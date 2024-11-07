# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:39:41 2024

@author: Usuario
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geo_northarrow import add_north_arrow
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
import scipy.io


# Cambia el directorio de trabajo
os.chdir(r'F:\buenaventura\Buenaventura_Map_Matlab\mapa')

# Carga de datos
buena_linea1_1_CORTO = np.loadtxt('buena_linea1_1_CORTO.dat')
outline = np.loadtxt('outline.dat')
Buenav_Isla1 = np.loadtxt('Buenav_Isla1.dat')
mascara = np.loadtxt('mascara.dat')
urbano = np.loadtxt('urbano.dat')
Batry = np.loadtxt('batimetria_malla.dat')


#carga el archivo de datos 
import h5py

# Ruta del archivo .h5
archivo = r'F:\buenaventura\archivo.h5'
AAAA=3
nsteps = 1000 
imageprefix = "velocity_"
for nt in range(10, nsteps):
    with h5py.File(archivo, 'r') as f:
        # Acceder al dataset 'mValues'
        ux  = f['mValues'][nt-1]  # Cargar todo el dataset en memoria
    
    vel_x = ux[3,:]
    vel_y = ux[4,:]
    velocidad = np.sqrt(vel_x**2 + vel_y**2)
    
    # Verificar si los datos están cargados correctamente
    if buena_linea1_1_CORTO.size == 0 or outline.size == 0:
        raise ValueError("Los archivos de datos están vacíos o no se pudieron cargar correctamente.")
    
    # Inicializar listas para almacenar las coordenadas
    latitudes = []
    longitudes = []
    
    # Cargar las coordenadas
    for data in [buena_linea1_1_CORTO, outline, Buenav_Isla1, urbano, mascara]:
        latitudes.extend(data[:, 1])  # Suponiendo que la columna 1 tiene las latitudes
        longitudes.extend(data[:, 0])  # Suponiendo que la columna 0 tiene las longitudes
    
    # Convertir listas a arrays de numpy
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    
    # Calcular límites
    lat_min = np.nanmin(latitudes)
    lat_max = np.nanmax(latitudes)
    lon_min = np.nanmin(longitudes)
    lon_max = np.nanmax(longitudes)
    
    # Variables de entrada
    scale = 2
    xxx = np.linspace(lon_min, lon_max, 1000)  # Cambiado para usar lon_min y lon_max
    yyy = np.linspace(lat_min, lat_max, 1000)  # Cambiado para usar lat_min y lat_max
    
    # Crear la malla
    X, Y = np.meshgrid(xxx, yyy)
    
    # Los datos de batimetría
    data = Batry[:, 1:3]  # Las coordenadas x, y
    z_values = velocidad  # Los valores Z de la batimetría
    
    # Interpolación de los datos
    Z = griddata(data, z_values, (X, Y), method='linear')  # Cambiado a 'linear' para suavizar
    
    # Filtrando los valores de z para aquellos que sean menores o iguales a -15
    outfal = np.where(z_values <= -15, z_values, np.nan)
    
    # Crear figura
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(zone=18))  
    levels = np.arange(0,1,0.025)
    # Graficar contornos
    hE = ax.contourf(X, Y, Z,  levels=levels, cmap='jet')
    
    XX=data[::AAAA, 0]
    YY =data[::AAAA, 1]
    xxvel =vel_x[::AAAA]
    yyvel =vel_y[::AAAA]
    Q = ax.quiver(XX, YY , xxvel,  yyvel, color="k", angles='xy',
          scale_units='xy', scale=.0005, width=.0005)
    qk = ax.quiverkey(Q, 0.5, 0.8, 0.5,  r'$0.2 \frac{m}{s}$',color="red" ,labelpos='E', labelcolor="red",
                   coordinates='figure')
   
    # ax.quiverkey(Q, 0.74, 0.18, 15, )
    # Cambiar el color de la flecha del quiverkey a rojo
    # qk._quiverkey.set_color('red')

# Cambiar el grosor de la flecha (line width) a 3 (ajustar según se necesite)
    # qk._quiverkey.set_linewidth(5)
    # Crear colorbar con la opción 'extend'
    # cbar = plt.colorbar(hE, orientation='vertical', extend='both')
    cbar_ax = fig.add_axes([0.905, 0.15, 0.025, 0.7])  # Ajuste del espacio a la izquierda
    cbar = plt.colorbar(hE, cax=cbar_ax, extend='both')
    # Etiqueta del colorba
    cbar.ax.set_ylabel('Velocity (m/s)')
    # Ajustar límites
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.UTM(zone=18))
    
    # # Dibujar el fondo del mapa
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.add_feature(cfeature.OCEAN, facecolor='aqua')
    
    # Graficar las capas
    ax.fill(buena_linea1_1_CORTO[:, 0], buena_linea1_1_CORTO[:, 1], color=[0.9, 0.9, 0.9])  # Línea buena gris
    ax.plot(buena_linea1_1_CORTO[:, 0], buena_linea1_1_CORTO[:, 1], linewidth=0.5, color='black')
    ax.fill(urbano[:, 0], urbano[:, 1], color="#9b8268")  # Rellenar urbano con color amarillo
    ax.fill(Buenav_Isla1[:, 0], Buenav_Isla1[:, 1], color="#9b8268")  # Rellenar urbano con color amarillo
    
    
    # Añadir otras islas
    for i in range(3, 26):
        isla_data = np.loadtxt(f'Buanv_islas_{i}.dat')
        ax.fill(isla_data[:, 0], isla_data[:, 1], color=[0.9, 0.9, 0.9])
        ax.plot(isla_data[:, 0], isla_data[:, 1], linewidth=0.5, color='black')
    
    # Añadir líneas de cuadrícula
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Títulos y anotaciones
    ax.text(2.755e+5, 4.31e+5, 'Buenaventura', fontsize=12, color='red', transform=ccrs.UTM(zone=18))
    
    
    # Ajustar los ejes
    ax.set_adjustable('box')  # Similar a axis tight en MATLAB
    
    # Configuración del mapa (localización en el geoide)
    new_axis_position = [0.68, 0.150, 0.20, 0.3]  # La posición del nuevo eje
    # Crea un nuevo eje para el mapa
    new_ax = fig.add_axes(new_axis_position, projection=ccrs.PlateCarree())
    
    # Mapa del mundo centrado en Colombia
    new_ax.set_extent([-85, -70, 0, 15], crs=ccrs.PlateCarree())
    new_ax.add_feature(cfeature.LAND, facecolor='lightgray')
    new_ax.add_feature(cfeature.OCEAN, facecolor='white')
    
    # Configura el color de fondo
    new_ax.set_facecolor([1, 1, 1])  # Color blanco para el océano
    
    # Añadir un punto (simulando geoshow)
    new_ax.plot(-77, 3.6, marker='+', color='red', markersize=10)
    
    # Añadir texto
    new_ax.text(-77, 2.53, 'Buenaventura (Col)', color='red', fontsize=7, fontweight='bold', ha='center')
    gl_new = new_ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.25, linestyle='--')
    gl_new.top_labels = False
    gl_new.right_labels = False
    
    
    
    # Añadir la flecha de norte
    add_north_arrow(ax, scale=.75, xlim_pos=.1, ylim_pos=.85, color='black', text_scaler=4, text_yT=-1.25)
    
    def meters_formatter(x, p):
        strRes = '{:,} Km'.format(int(x/1000)) 
        return strRes
    
    
    def scaleBar(x,y,mapdistance,ax,trans,subdivision=1,height=.02):
        """x - lower left corner of arrow in trans coordinates
           y - lower left corner of arrow in trans coordiantes
           mapdistance - maximum distance to show on the scalebar
           ax - axes to add patch and text
           trans - transformation the coordinates are in
           subdivision - number of subdivisions to show in the scalebar
           height - height of the bar part of the scalebar"""
        xmin, xmax = ax.get_xlim() #returns left,right
        abs_width = abs(xmax-xmin)
        length = 1.0/abs_width * mapdistance
        if subdivision > 1.0:
            sublength = float(length)/subdivision
            fColor = 'black'
            subx = x
            for i in range(0,subdivision):
    
                ax.add_patch(mpatches.Rectangle((subx,y), sublength, height, transform=trans,facecolor=fColor,edgecolor='black',lw=.5))
                subx += sublength
                if fColor == 'black':
                    fColor = 'white'
                else:
                    fColor = 'black'
                
        else:
            ax.add_patch(mpatches.Rectangle((x,y), length, height, transform=trans,facecolor='black',edgecolor='black'))
            
        ax.text(x,y+height*1.5,'0',transform=trans,ha='center')
        ax.text(x+length,y+height*1.5,meters_formatter(mapdistance,None),transform=trans,ha='center')
    
    
    scaleBar(0.051,.025,10000,ax,ax.transAxes,subdivision=4)
    
    # Crear la carpeta New_Map si no existe
    output_dir = 'New_Map'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Guardar la figura como un archivo PNG
    output_dir = r'F:\buenaventura\Buenaventura_Map_Matlab\mapa\New_Map'
    # os.path.join(output_dir)
    # output_path = os.path.join(output_dir, 'Velocity_Buenaventura.png')
    # plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(os.path.join(output_dir, f"{imageprefix}{nt:04d}.jpg"), dpi=300)
    # Mostrar el mapa
    plt.show()
