import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
model_type = "Temporal"
data_set = "Validation_data"


try:
    os.mkdir("../../" + data_set +"_statistics/plots")
except FileExistsError as fe:
    pass
try:
    os.mkdir("../../" + data_set +"_statistics/plots/" + model_type)
except FileExistsError as fe:
    pass

path = "../../" + data_set +"_statistics/plots/" + model_type
print("Loading config and generator...")
if model_type == "Spatiotemporal":
    spatiotemporal = True
    from Spatiotemporal.data_configuration import Config
    from Spatiotemporal.batch_generator import BatchGenerator

elif model_type == "Temporal":
    temporal = True
    from Temporal.data_configuration import Config
    from Temporal.batch_generator import BatchGenerator

else:
    spatial = True
    from Spatial.data_configuration import Config
    from Spatial.batch_generator import BatchGenerator

conf = Config()
conf.filter_input = True
conf.upsample_input = True
conf.random_validation_sampling = True
conf.train_conf.batch_size = 16

g = BatchGenerator(conf, data=data_set)
input_measures = [key for key in conf.available_columns if conf.input_data[key]]
output_measures = [key for key in conf.available_columns if conf.output_data[key]]

print("Length of generator: " + str(len(g)))
brake_counter = Counter()
steer_counter = Counter()
direction_counter = Counter()
speed_counter = Counter()
speed_limit_counter = Counter()
traffic_light_counter = Counter()
for b in range(len(g)):
    #if b % 100 == 0:
    #r = np.random.randint(0, len(g))
    print("\r Progress: " + str(100*b/len(g)), end="")
    n = g[b]
    bx = n[0]
    by = n[1]
    #print(bx)
    for i in range(conf.train_conf.batch_size):
        speed = bx["input_Speed"][i][-1]
        if speed < conf.filter_threshold_speed:
            speed_counter[0] += 1
        else:
            speed_counter[1] += 1

        speed_limit = np.where(bx["input_ohe_speed_limit"][i][0] == 1)[0][0]
        speed_limit_counter[speed_limit] += 1

        traffic_light = np.where(bx["input_TL_state"][i][0] == 1)[0][0]
        traffic_light_counter[traffic_light] += 1

        direction = np.where(bx["input_Direction"][0][0] == 1)[0][0]
        direction_counter[direction] += 1

        brake = int(by["output_Brake"][i][0])
        brake_counter[brake] += 1

        steer = int(by["output_Steer"][i][0]*10)
        steer_counter[steer] += 1

def plot_piechart(counted, label_names, colors, title):
    keys = list(counted.keys())
    values = list(counted.values())
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    s = sum(values)
    labels = []
    percentages = []
    for key in keys:
        percentages.append(str(np.round((float(counted.get(key))/s)*100, 2)) + "%")
        labels.append(str(label_names[key]))
    wedges, autotexts = ax.pie(values, labels=percentages, colors=colors)

    ax.legend(wedges, labels,
            #title="Labels",
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    plt.tight_layout()
    
    ax.set_title(" ".join(title.split("_")[2:]), pad=-5)
    plt.savefig(path + "/" +title)


def plot_histogram(counted, title):
    fig, ax = plt.subplots()
    labels = list(counted.keys())
    ax.bar([float(x)/10 for x in list(counted.keys())], list(counted.values()), width=0.1)
    ax.set_title(" ".join(title.split("_")[2:]))
    fig.savefig(path + "/" + title)

dir_labels = ["Lanefollow", "Left", "Right", "Straight"]
tl_labels = ["Green", "Red", "Yellow"]
brake_labels = ["No braking", "braking"]
sl_labels = ["30 km/h", "60 km/h", "90 km/h"]
four_colors =  ["#003f5c", "#7a5195", "#ef5675", "#ffa600"]
three_colors =  ["#003f5c", "#bc5090", "#ffa600"]
two_colors  =  ["#003f5c", "#ffa600"]
tl_color = ["Green", "Red", "Yellow"]
speed_labels = ["Standing still", "Driving"]
plot_piechart(traffic_light_counter, tl_labels, three_colors, "pie_chart_Traffic_light_distribution")
plot_piechart(speed_limit_counter, sl_labels, three_colors, "pie_chart_Speed_limit_distribution")
plot_piechart(direction_counter, dir_labels, four_colors, "pie_chart_Direction_distribution")
plot_piechart(brake_counter, brake_labels, two_colors, "pie_chart_Brake_distribution")
plot_piechart(speed_counter, speed_labels, two_colors, "pie_chart_Speed_distribution")

plot_histogram(steer_counter, "histogram_Steering_distribution")


f = open(path + "/counters.txt", "w+")
f.write("Total amount of samples: " + str(len(g)) + "\n")
f.write("steer_counter" + str(steer_counter) + "\n")
f.write("direction_counter" + str(direction_counter) + "\n")
f.write("brake_counter" + str(brake_counter) + "\n")
f.write("speed_limit_counter" + str(speed_limit_counter) + "\n")
f.write("traffic_light_counter" + str(traffic_light_counter) + "\n")
f.close()