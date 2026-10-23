import xarray as xr
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def procesar_dia(args):
    """Función auxiliar para procesar un solo día en un núcleo independiente."""
    fecha_str, imerg_dir, daily_dir = args
    daily_output = os.path.join(daily_dir, f"imerg_{fecha_str}.nc")
    
    if os.path.exists(daily_output):
        return daily_output
        
    patron = os.path.join(imerg_dir, f"3B-HHR.MS.MRG.3IMERG.{fecha_str}*.nc*")
    archivos_dia = sorted(glob.glob(patron))
    
    if not archivos_dia:
        return None
        
    try:
        datasets = [xr.open_dataset(f) for f in archivos_dia]
        ds_dia = xr.concat(datasets, dim="time")
        ds_dia.to_netcdf(daily_output)
        
        for ds in datasets:
            ds.close()
        ds_dia.close()
        
        return daily_output
    except Exception as e:
        print(f"❌ Error en el día {fecha_str}: {e}")
        return None

def unificar_imerg_paralelo(data_dir):
    imerg_dir = os.path.join(data_dir, "imerg_2024_raw")
    daily_dir = os.path.join(imerg_dir, "daily_temp")
    os.makedirs(daily_dir, exist_ok=True)
    
    dias_por_mes = {
        "01": 31, "02": 29, "03": 31, "04": 30,
        "05": 31, "06": 30, "07": 31, "08": 31,
        "09": 30, "10": 31, "11": 30, "12": 31
    }
    
    for mes_str, total_dias in dias_por_mes.items():
        print(f"\n📅 Procesando mes 2024-{mes_str} en paralelo...")
        
        # Preparar tareas para todos los días del mes
        tareas = []
        for dia in range(1, total_dias + 1):
            fecha_str = f"2024{mes_str}{dia:02d}"
            tareas.append((fecha_str, imerg_dir, daily_dir))
            
        daily_files = []
        
        # Usar todos los núcleos disponibles de la CPU (max_workers=None usa todos)
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(procesar_dia, tarea): tarea[0] for tarea in tareas}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Mes 2024-{mes_str}"):
                res = future.result()
                if res:
                    daily_files.append(res)
                    
        # Fase Mensual: Unir los diarios resultantes
        if daily_files:
            print(f"📦 Unificando días en el archivo mensual definitivo para 2024-{mes_str}...")
            try:
                # Ordenar cronológicamente
                daily_files = sorted(daily_files)
                datasets_mes = [xr.open_dataset(f) for f in daily_files]
                ds_mes = xr.concat(datasets_mes, dim="time")
                
                output_monthly = os.path.join(imerg_dir, f"imerg_complete_2024_{mes_str}.nc")
                ds_mes.to_netcdf(output_monthly)
                
                for ds in datasets_mes:
                    ds.close()
                ds_mes.close()
                
                print(f"✅ ¡Guardado con éxito: {output_monthly}!")
                
            except Exception as e:
                print(f"❌ Error unificando el mes {mes_str}: {e}")

    print("\n🎉 ¡Proceso paralelo completado!")

if __name__ == "__main__":
    data_path = "data/raw/"
    unificar_imerg_paralelo(data_path)