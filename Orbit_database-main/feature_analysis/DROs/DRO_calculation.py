import requests


url1 = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api?sys=jupiter-europa&family=dro"

data = requests.get(url1).json()
interpolate = float(data["data"][3974][4])+(1.0306454520330-float(data["data"][3974][0]))/(float(data["data"][3973][0])-float(data["data"][3974][0]))*(float(data["data"][3973][4])-float(data["data"][3974][4]))
interpolate_2 = float(data["data"][3974][7])+(1.0306454520330-float(data["data"][3974][0]))/(float(data["data"][3973][0])-float(data["data"][3974][0]))*(float(data["data"][3973][7])-float(data["data"][3974][7]))

print("Europa DRO v2 at r1=1.0306454520330: "+str(interpolate))
print("Europa DRO with r1=1.0306454520330 orbital period: "+str(interpolate_2))

url2 = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api?sys=earth-moon&family=dro"
data2 = requests.get(url2).json()
interpolate_3 = float(data2["data"][2946][4])+(1.1179191776797255-float(data2["data"][2948][0]))/(float(data2["data"][2946][0])-float(data2["data"][2947][0]))*(float(data2["data"][2946][4])-float(data2["data"][2947][4]))
interpolate_4 = float(data2["data"][2946][7])+(1.1179191776797255-float(data2["data"][2948][0]))/(float(data2["data"][2946][0])-float(data2["data"][2947][0]))*(float(data2["data"][2946][7])-float(data2["data"][2947][7]))
print("Moon DRO v2 at r1=1.1179191776797255: NOT IN DATABASE"+str(interpolate_3))
print("Moon DRO with r1=1.1179191776797255 orbital period: "+str(interpolate_4))