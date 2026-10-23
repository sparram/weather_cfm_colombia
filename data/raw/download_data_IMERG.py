import os
import time
import calendar
import earthaccess
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def obtener_nombre_salida(granule):
    """Extrae el nombre de salida esperado para un gránulo"""
    s3_url = ""
    for link in granule.get("umm", {}).get("RelatedUrls", []):
        url = link.get("URL", "")
        if url.startswith("s3://"):
            s3_url = url
            break
    if not s3_url:
        # Fallback usando el id o título si no hay s3_url directa
        s3_url = granule.get("title", "unknown")
    nombre_archivo = s3_url.split('/')[-1]
    return nombre_archivo.replace(".HDF5", "_colombia.nc4").replace(".h5", "_colombia.nc4")

def procesar_archivo_local(archivo_path, min_lon, min_lat, max_lon, max_lat, path_destino):
    try:
        if not isinstance(archivo_path, str):
            archivo_path = str(archivo_path)
            
        nombre_archivo = os.path.basename(archivo_path)
        nombre_salida = nombre_archivo.replace(".HDF5", "_colombia.nc4").replace(".h5", "_colombia.nc4")
        output_filepath = os.path.join(path_destino, nombre_salida)
        
        if os.path.exists(output_filepath):
            if os.path.exists(archivo_path):
                os.remove(archivo_path)
            return "Ya existe"
            
        with xr.open_dataset(archivo_path, group='Grid') as ds:
            ds_recorte = ds.sel(
                lat=slice(min_lat, max_lat), 
                lon=slice(min_lon, max_lon)
            )
            ds_recorte.to_netcdf(output_filepath)
            
        if os.path.exists(archivo_path):
            os.remove(archivo_path)
            
        return "OK"
    except Exception as ex:
        if os.path.exists(archivo_path):
            try:
                os.remove(archivo_path)
            except:
                pass
        return f"Error en {os.path.basename(archivo_path)}: {ex}"

def buscar_con_reintentos(short_name, version, temporal, bounding_box, max_intentos=5):
    for intento in range(1, max_intentos + 1):
        try:
            resultados = earthaccess.search_data(
                short_name=short_name,
                version=version,
                temporal=temporal,
                bounding_box=bounding_box
            )
            return resultados
        except Exception as e:
            print(f"\n⚠️ Advertencia de red en búsqueda (Intento {intento}/{max_intentos}): {e}")
            if intento == max_intentos:
                raise e
            time.sleep(5 * intento)

def descargar_con_reintentos(resultados, path_temporal, max_intentos=5):
    for intento in range(1, max_intentos + 1):
        try:
            archivos = earthaccess.download(resultados, path_temporal)
            return archivos
        except Exception as e:
            print(f"\n⚠️ Advertencia de red en descarga (Intento {intento}/{max_intentos}): {e}")
            if intento == max_intentos:
                raise e
            time.sleep(5 * intento)

def procesar_imerg_por_meses_inteligente():
    print("🔐 Autenticándose en NASA Earthdata...")
    earthaccess.login(persist=True)
    
    min_lon, min_lat, max_lon, max_lat = -82.0, -4.5, -66.75, 12.75
    anio = 2025
    meses_a_procesar = range(1, 13)
    
    path_temporal = "./temp_gpm_downloads"
    path_destino = "./datos_imerg_colombia_s3"
    os.makedirs(path_temporal, exist_ok=True)
    os.makedirs(path_destino, exist_ok=True)
    
    max_workers = 8
    
    for mes in meses_a_procesar:
        ultimo_dia = calendar.monthrange(anio, mes)[1]
        fecha_inicio = f"{anio}-{mes:02d}-01"
        fecha_fin = f"{anio}-{mes:02d}-{ultimo_dia}"
        
        print(f"\n==================================================")
        print(f"📅 Analizando periodo: {fecha_inicio} al {fecha_fin}")
        print(f"==================================================")
        
        try:
            resultados = buscar_con_reintentos(
                short_name='GPM_3IMERGHH',
                version='07',
                temporal=(f"{fecha_inicio}T00:00:00Z", f"{fecha_fin}T23:59:59Z"),
                bounding_box=(min_lon, min_lat, max_lon, max_lat)
            )
        except Exception as e:
            print(f"❌ Error buscando datos para el mes {mes}: {e}")
            continue
        
        print(f"📦 Se encontraron {len(resultados)} gránulos en total para este mes en la NASA.")
        
        # FILTRO INTELIGENTE: Descartar los gránulos que ya tienen su archivo recortado localmente
        resultados_pendientes = []
        for g in resultados:
            nombre_salida = obtener_nombre_salida(g)
            output_filepath = os.path.join(path_destino, nombre_salida)
            if not os.path.exists(output_filepath):
                resultados_pendientes.append(g)
                
        print(f"🔍 Pendientes por descargar/procesar: {len(resultados_pendientes)}/{len(resultados)}")
        
        if len(resultados_pendientes) == 0:
            print(f"✨ ¡El mes {mes:02d} ya está 100% procesado! Saltando al siguiente...")
            continue
            
        print("📥 Fase 1: Descargando gránulos pendientes en paralelo...")
        try:
            archivos_locales = descargar_con_reintentos(resultados_pendientes, path_temporal)
        except Exception as e:
            print(f"❌ Error descargando el mes {mes}: {e}")
            continue
            
        if not archivos_locales:
            continue
            
        print("✂️ Fase 2: Recortando y limpiando en paralelo localmente...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(procesar_archivo_local, ap, min_lon, min_lat, max_lon, max_lat, path_destino)
                for ap in archivos_locales
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Mes {mes:02d}"):
                res = future.result()
                if res and "Error" in res:
                    print(f"\n⚠️ {res}")

    try:
        os.rmdir(path_temporal)
    except OSError:
        pass

    print("\n🎉 ¡Proceso anual completado con éxito y de forma ultra eficiente!")

if __name__ == "__main__":
    procesar_imerg_por_meses_inteligente()