import streamlit as st
import pandas as pd
import numpy as np
#import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point,Polygon,MultiPolygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Voronoi, voronoi_plot_2d
#import folium as fl

plt.style.use("ggplot")

# ### Importacao dos arquivos ### #
def get_data1():
    path = "data/final_novo_ibge_sp.csv"
    return pd.read_csv(path)

def get_data2():
    path = "data/Populacao_Densidade.cvs"
    return pd.read_csv(path)

def get_shape():
    path = "data/SP_Municipios_2020.shp"
    return gpd.read_file(path)

def get_postos():
    path = "data/ubs_sp_com_pontos.csv"
    return  pd.read_csv(path)

# ## Nome da Apicacao ### #
st.title("Sistema de Recomendação para Cidades Aumentarem o Fluxo de Vacinação")
#st.write('Orientações para combater o COVID')
st.subheader("Para visualizar o mapa inteiro da cidade de São Paulo, [clique aqui](mapas/mapa_geral.html). ")

# ### Craicao das sidebars ### #
st.sidebar.header('Escolha sua cidade:')

ibge = get_data1()
pDensidade = get_data2()
mapas = get_shape()

cidade = ibge['Município']
cidade_choice = st.sidebar.selectbox('Cidade', cidade)

# ### Cidade Mais Similares ### #
cod_ibge = get_data2().columns[0]

top_csv = np.zeros((1,6),dtype=int)

for city in range(0,pDensidade .shape[0]): # para cada cidade
  actual_city = pDensidade .iloc[city]
  #print(cod_ibge)
  search = []
  for line in range(1,actual_city.shape[0]): # crio o dicionáário
      cod_ibge = str(pDensidade .columns[line])
      dist = float(actual_city[line])
      search.append((cod_ibge, dist))

search.sort(key=lambda x:x[1]) # ordeno pela distancia
top5 = search[1:6] # pegar top5 cidades mais próximas
regist = np.array([[int(actual_city[0]),int(top5[0][0]),int(top5[1][0]),int(top5[2][0]),int(top5[3][0]),int(top5[4][0]) ]] )
top_csv = np.concatenate((top_csv,regist))

dataset = pd.DataFrame()
dataset['Cidade escolhida'] = top_csv[:,0]
dataset['Cidade mais similiar'] = top_csv[:,1]
dataset['2° cidade mais similiar'] = top_csv[:,2]
dataset['3° cidade mais similiar'] = top_csv[:,3]
dataset['4° cidade mais similiar'] = top_csv[:,4]
dataset['5° cidade mais similiar'] = top_csv[:,5]
dataset.drop(0,inplace=True)

for index, i in mapas.iterrows():
  mapas.loc[index, 'NM_MUN'] = str(i.NM_MUN).upper()

map = mapas[mapas['NM_MUN'] == cidade_choice]
cidade_ibge = ibge[ibge['Município'] == cidade_choice]

#Informações da Cidade
codigo_ibge = cidade_ibge['Codigo_IBGE'].values
populacao = cidade_ibge['População Estimada'].values
st.subheader("Informações sobre a cidade de {}".format(str(cidade_choice)))
st.write(cidade_ibge.iloc[:,0:12])

#Postos na cidade
postos = get_postos()
postos = postos.dropna()
codigo_ibge = int(codigo_ibge)
st.write(codigo_ibge)
codigo_ibge = str(codigo_ibge)
codigo_ibge = codigo_ibge[:-1]
codigo_ibge = int(codigo_ibge)

n_postos_cidade = postos[postos['IBGE'] == codigo_ibge]
st.write("**Número de postos na cidade, dentro e fora do território:** {} ".format(n_postos_cidade.shape[0]))

postos['geometry'] = None
for index, i in postos.iterrows():
  postos.loc[index, 'geometry'] = Point(i.LONGITUDE, i.LATITUDE)

gdf_postos = gpd.GeoDataFrame(postos, geometry='geometry')
gdf_postos.crs = "epsg:4326"

poly_map = map.iloc[0].geometry
postos_cidade= gdf_postos[gdf_postos.intersects(poly_map)]

fig, ax = plt.subplots(figsize=(10, 8))
postos_cidade.plot(ax=ax)
map.plot(ax=ax, facecolor='None', edgecolor="black")
plt.xticks(rotation=45)
plt.show()
plt.savefig("img/map_postos.png")

#Mapa
map.plot()
plt.xticks(rotation=45)
plt.show()
plt.savefig("img/map.png")

#Plota Mapa
st.subheader("Mapa da cidade de {}".format(str(cidade_choice)))
st.image("img/map.png")

st.subheader("Mapa da cidade {}, marcados com as Unidades Básicas de Saúde dentro do território do município.".format(str(cidade_choice)))
st.image("img/map_postos.png")


def voronoi_finite_polygons_2d(vor, radius=None):

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

pop_ubs = populacao/n_postos_cidade.shape[0]

#Eixo convexo
if n_postos_cidade.shape[0] > 2:
    coords = ['LATITUDE', 'LONGITUDE']
    postos_coords_cidade = postos_cidade[coords]
    postos_coords_cidade = np.asarray(postos_coords_cidade)

    hull = ConvexHull(postos_coords_cidade)

    _ = convex_hull_plot_2d(hull)
    #plt.show()
    plt.savefig("img/convex_hull.png")

    st.subheader("Eixo convexo formado pelas UBS da cidade {}".format(str(cidade_choice)))
    st.write("**Número de habitantes para cada UBS:** {}".format(populacao/n_postos_cidade.shape[0]))
    st.write("**Área do Eixo convexo:** {}".format(hull.area))
    st.image("img/convex_hull.png")
    area_eixo = hull.area

    st.subheader("Diagrama de Voronoi formado pelas UBS da cidade:")
    vor = Voronoi(postos_coords_cidade)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)

    plt.plot(postos_coords_cidade[:, 0], postos_coords_cidade[:, 1], 'ko')
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.05)
    #plt.show()
    plt.savefig("img/diagrama_voronoi.png")

    st.image("img/diagrama_voronoi.png")

elif n_postos_cidade.shape[0] == 2:
    st.write("Sua cidade possui apenas **{}** UBS, com isso temos cerca de **{}** habitantes para cada uma".format(n_postos_cidade[0], populacao/n_postos_cidade.shape[0]))

elif n_postos_cidade.shape[0] == 1:
    st.write("Sua cidade possui apenas **1** Unidade Básica de Saúde.")

else:
    st.write("Sua cidade **não possui nenhuma** Unidade Básica de Saúde.")

#===========================
#Mapa interativo:

#st.subheader("Mapa:")
#for index, i in mapas.iterrows():
 #   qtd_postos = len(postos[gdf_postos.intersects(i.geometry)])
 #   mapas.loc[index, 'qtd_postos'] = qtd_postos

#media_lat = gdf_postos['LATITUDE'].mean()
#media_log = gdf_postos['LONGITUDE'].mean()

#mapa = fl.Map(location=[media_lat, media_log], zoom_start=9)

#for _, i in mapas.iterrows():
    #municipio_geojson = fl.features.GeoJson(i.geometry,
      #                                      style_function=lambda feature: {
      #                                          'color': 'purple',
     #                                           'weight': 2,
    #                                            'fillOpacity': 0.1
   #                                         })


  #  popup = fl.Popup("""
 #                 Municipio: {}
 #                Quantidade postos: {}
 #               """.format(i.NM_MUN, str(int(i.qtd_postos))))

#    popup.add_to(municipio_geojson)
#    municipio_geojson.add_to(mapa)

#mapa.save("mapas/mapa_geral.html")

##st.components.v1.iframe("mapas/mapa_geral.html")
#st.write(mapa)



#media_lat = postos_cidade['LATITUDE'].mean()
#media_log = postos_cidade['LONGITUDE'].mean()

#mapa = fl.Map(location=[media_lat, media_log], zoom_start=9)

#municipio_geojson = fl.features.GeoJson(postos_cidade.geometry,
     #                                   style_function=lambda feature: {
   #                                         'color': 'purple',
   #                                         'weight': 2,
  #                                              'fillOpacity': 0.1
 #                                       })

#popup = fl.Popup("Municipio: {}"
#                "Quantidade postos: {}".format(cidade_choice, n_postos_cidade[0]))

#popup.add_to(municipio_geojson)
#municipio_geojson.add_to(mapa)

#mapa.save("mapas/mapa_cidade.html")
#folium_static(mapa)

#st.components.v1.iframe("mapas/mapa_cidade.html")
#st.write(mapa)

#========================== Voronoi

st.header("Recomendações")
st.subheader("Confira as 5 cidades mais parecidas com a cidade que voce escolheu: ")
st.write(dataset)

cidades_proximas = top_csv.tolist()
#st.write(cidades_proximas)
cidades_proximas = cidades_proximas[1]

for i in range(len(cidades_proximas)):
    cidades_proximas[i] = str(cidades_proximas[i])

cidade1 = ibge[ibge['Codigo_IBGE'] == cidades_proximas[1]]
st.write(cidade1)

if n_postos_cidade.shape[0] > 2:
    if area_eixo <= 0.5:
        st.write("As Unidades Básicas de Saúde da sua cidade poderiam estar mais espalhadas,"
                 " assim elas atenderiam uma população maior e possivelmente rural.")
st.write("\n")

if pop_ubs >= 10000:
    st.write("Essa cidade possuí mais de 10 mil habitantes para cada UBS, isso pode acarretar problemas futuros. "
        "Com um número maior de unidades básicas de saúde, a velocidade de vacinação municipio iria aumentar,"
        " consequentemente, o número de mortes viria a cair.")