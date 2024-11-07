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
import matplotlib.lines as mlines
from matplotlib.patches import Polygon
# Cambia el directorio de trabajo
os.chdir(r'F:\buenaventura\Buenaventura_Map_Matlab\mapa')

# Carga de datos
buena_linea1_1_CORTO = np.loadtxt('buena_linea1_1_CORTO.dat')
outline = np.loadtxt('outline.dat')
Buenav_Isla1 = np.loadtxt('Buenav_Isla1.dat')
mascara = np.loadtxt('mascara.dat')
urbano = np.loadtxt('urbano.dat')
malla = np.loadtxt('malla.dat')
Tmalla = np.array(malla[:, [1, 3, 5]], dtype=int) 


Pmalla = np.loadtxt('Pmalla.dat')
P = Pmalla[:, [1, 2]]

# Verificar si los datos están cargados correctamente
if buena_linea1_1_CORTO.size == 0 or outline.size == 0:
    raise ValueError("Los archivos de datos están vacíos o no se pudieron cargar correctamente.")

# Inicializar listas para almacenar las coordenadas
latitudes = []
longitudes = []

# Cargar las coordenadas de los archivos relevantes
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

# Crear figura y definir proyección adecuada para tus datos
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.UTM(zone=18))  # Reemplaza con la zona adecuada

# Ajustar límites
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.UTM(zone=18))

# # Dibujar el fondo del mapa
# ax.add_feature(cfeature.OCEAN)

# Graficar las capas
ax.fill(buena_linea1_1_CORTO[:, 0], buena_linea1_1_CORTO[:, 1], color=[0.9, 0.9, 0.9])  # Línea buena gris
ax.plot(buena_linea1_1_CORTO[:, 0], buena_linea1_1_CORTO[:, 1], linewidth=0.5, color='black')  
ax.fill(urbano[:, 0], urbano[:, 1], color=[1, 1, 0.2])  # Rellenar urbano con color amarillo
ax.fill(Buenav_Isla1[:, 0], Buenav_Isla1[:, 1], color=[1, 1, 0.2])  # Isla color amarillo
ax.plot(Buenav_Isla1[:, 0], Buenav_Isla1[:, 1], linewidth=0.5, color='black')  # Contorno de la isla

# Añadir otras islas
for i in range(3, 26):
    isla_data = np.loadtxt(f'Buanv_islas_{i}.dat')
    ax.fill(isla_data[:, 0], isla_data[:, 1], color=[0.9, 0.9, 0.9])  # Línea buena gris
    ax.plot(isla_data[:, 0], isla_data[:, 1], linewidth=0.5, color='black')  # Contorno de la isla


for tri in Tmalla:
    # Obtener las coordenadas de los vértices del triángulo
    vertices = P[tri-1, :]
    polygon = plt.Polygon(vertices, facecolor='none', edgecolor='blue', linewidth=0.4, alpha=0.5)  # Sin color de fondo
    ax.add_patch(polygon)

# Graficar los puntos
ax.plot(P[:, 0], P[:, 1], 'o', color='red', markersize=.0005)  # Vértices en rojo


# Graficar los puntos
# ax.plot(P[:, 0], P[:, 1], 'o', color='green', markersize=0.05)

# Añadir líneas de cuadrícula
gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Títulos y anotaciones
# ax.text(2.45e+5, 4.2e+5, 'Pacific Ocean', fontsize=8, transform=ccrs.UTM(zone=18))
ax.text(2.755e+5, 4.31e+5, 'Buenaventura', fontsize=12, color='red', transform=ccrs.UTM(zone=18))
# ax.text(2.6e+5, 4.25e+5, 'Buenaventura\nBay', fontsize=8, color='blue', transform=ccrs.UTM(zone=18))

# # Añadir marcadores
# ax.plot(2.57e+5, 4.23e+5, 'mo', markersize=5, markeredgecolor='black', markerfacecolor='black', transform=ccrs.UTM(zone=18))
# ax.plot(2.68e+5, 4.29e+5, 'mo', markersize=5, markeredgecolor='black', markerfacecolor='black', transform=ccrs.UTM(zone=18))
# ax.text(2.575e+5, 4.23e+5, 'T1', fontsize=7, color='black', transform=ccrs.UTM(zone=18))
# ax.text(2.68e+5, 4.28e+5, 'T2', fontsize=7, color='black', transform=ccrs.UTM(zone=18))

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

# Añadir la flecha de norte al nuevo eje
add_north_arrow(new_ax, scale=.75, xlim_pos=.280, ylim_pos=.811, color='black', text_scaler=4, text_yT=-1.25)

# Crear la carpeta New_Map si no existe
output_dir = 'New_Map'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar la figura como un archivo BMP
output_path = os.path.join(output_dir, 'Malla_Buenaventura.bmp')
plt.savefig(output_path, format='tiff', dpi=300)

# Mostrar el mapa
plt.show()
