import os
import xarray as xr

def procesar_dem_superficie(data_dir):
    input_path = os.path.join(data_dir, "surface_geopotential_era5.nc")
    
    if not os.path.exists(input_path):
        print(f"❌ No se encontró el archivo de superficie: {input_path}")
        return
        
    print(f"📥 Procesando la orografía desde: {input_path}")
    ds = xr.open_dataset(input_path)
    
    # Verificamos el nombre de la variable (en single levels suele ser 'z')
    var_name = 'z' if 'z' in ds.data_vars else list(ds.data_vars)[0]
    
    # Extraemos el único tiempo disponible y convertimos geopotencial (m^2/s^2) a metros (m)
    dem_surface = ds[var_name].isel(valid_time=0) / 9.80665
    
    output_path = os.path.join(data_dir, "dem_colombia_era5.nc")
    
    # Creamos un dataset limpio con la variable 'elevation'
    ds_dem = xr.Dataset({"elevation": dem_surface})
    ds_dem.to_netcdf(output_path)
    
    print(f"✅ ¡DEM superficial procesado y guardado en: {output_path}!")
    print(f"   - Dimensiones del relieve: {dem_surface.shape} (Lat, Lon)")
    print(f"   - Altura máxima (Cumbres de los Andes): {float(dem_surface.max()):.1f} m")
    print(f"   - Altura mínima (Nivel del mar / Costas): {float(dem_surface.min()):.1f} m")
    
    ds.close()

if __name__ == "__main__":
    data_path = "data/processed/"
    procesar_dem_superficie(data_path)