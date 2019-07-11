import pandas as pd
import numpy as np
pd.set_option("display.max_rows", 200)

model_type = "Spatial"
current_model_nr = 9
df1 = pd.read_csv("Test_recordings/" + model_type + "/model-" + str(current_model_nr) + "/results.csv")

model_type = "Spatiotemporal"
current_model_nr = 20
df2 = pd.read_csv("Test_recordings/" + model_type + "/model-" + str(current_model_nr) + "/results.csv")
completed = df2.groupby(["Completion"])
completed_count = completed.count().loc["target_reached", "Loss"]
print(completed_count)
exit()
model_type = "Temporal"
current_model_nr = 25
df3 = pd.read_csv("Test_recordings/" + model_type + "/model-" + str(current_model_nr) + "/results.csv")
dataframes = [df1, df2, df3]

df = pd.concat([df1, df2, df3], ignore_index=True)

nets = df.groupby(["Network"])
mean_values = nets.mean().loc[:, ["Crossed_line_count", "Distance", "Speed", "Steer", "Model", "Epoch", "Loss"]]
mean_values.to_csv("Final_results/mean_values.csv", index=True)

# Group test results by weather
weather = df.groupby(["Completion", "Weather", "Network"])
weather_count = weather.count().loc["target_reached", "Loss"]
weather_count = pd.DataFrame(weather_count)
weather_count = weather_count.unstack(level=1)
weather_count.columns.set_levels([""], level=0, inplace=True)
weather_count.to_csv("Final_results/weather_based_completion.csv", index=True)

#completion2 = df.groupby(["Network", "Weather"])
#print(completion2.mean().loc[:, ["Distance", "Steer", "Speed"]])

# Group test results by Cars
cars = df.groupby(["Completion", "Cars", "Network"])
cars_count = cars.count().loc["target_reached", "Loss"]
cars_count = pd.DataFrame(cars_count).T
cars_count = cars_count.rename({"Loss": "Targets reached"}, axis='index')
cars_count.columns.set_levels(['No', 'Yes'], level=0, inplace=True)
#cars_count = cars_count.swaplevel(i=0, j=1,axis=1)
cars_count.to_csv("Final_results/cars_based_completion.csv", index=True)
#completion2 = df.groupby(["Network", "Cars"])
#print(completion2.mean().loc[:, ["Distance", "Steer", "Speed"]])

# Group test results by track
tracks = df.groupby(["Completion", "Track", "Network"])
track_count = tracks.count().loc["target_reached", "Loss"]
track_count = pd.DataFrame(track_count)
track_count = track_count.unstack(level=1)
track_count.columns.set_levels([""], level=0, inplace=True)
track_count.to_csv("Final_results/track_based_completion.csv", index=True)
#completion2 = df.groupby(["Network", "Track"])
#print(completion2.mean().loc[:, ["Distance", "Steer", "Speed"]])

# Group test results by track
completed = df.groupby(["Completion", "Network"])
completed_count = completed.count().loc["target_reached", "Loss"].values
size_count = [len(df1), len(df2), len(df3)]
percentage = [
    str(np.round(100*float(completed_count[0])/size_count[0],1)) + "%",
    str(np.round(100*float(completed_count[1])/size_count[1],1)) + "%",
    str(np.round(100*float(completed_count[2])/size_count[2],1)) + "%",
]
targets_reached = pd.DataFrame(
    data=[size_count, completed_count, percentage],
    index=["Test_episodes", "Targets_reached", "Completion degree"],
    columns=["Spatial", "Spatiotemporal", "Temporal"]
)

targets_reached.to_csv("Final_results/count_values.csv", index=True)
"""
for dataframe in dataframes:
    # Group test results by weather
    weather = dataframe.groupby(["Completion", "Network", "Weather"])
    weather_count = weather.count().loc["target_reached", "Loss"]

    #completion2 = df.groupby(["Network", "Weather"])
    #print(completion2.mean().loc[:, ["Distance", "Steer", "Speed"]])

    # Group test results by Cars
    cars = dataframe.groupby(["Completion", "Network", "Cars"])
    cars_count = cars.count().loc["target_reached", "Loss"]
    #completion2 = df.groupby(["Network", "Cars"])
    #print(completion2.mean().loc[:, ["Distance", "Steer", "Speed"]])

    # Group test results by track
    tracks = dataframe.groupby(["Completion", "Network", "Track"])
    track_count = tracks.count().loc["target_reached", "Loss"]
    #completion2 = df.groupby(["Network", "Track"])
    #print(completion2.mean().loc[:, ["Distance", "Steer", "Speed"]])
"""