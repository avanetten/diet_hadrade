![Alt text](/figs/header.png?raw=true "")

# Diet Hadrade

Humanitarian crises, along with both natural and man-made disasters show no sign of abating in the near term given ongoing changes to the planet and society. Yet ever improving machine learning techniques hold the possibility of blunting some of the harshest impacts of future crises.  Diet Hadrade provides a lightweight mechanism for leveraging remote sensing imagery and machine learning techniques to aid in humanitarian assistance and disaster response (HADR) in in austere environments. In a disaster scenario where communications are unreliable, overhead imagery often provides the first glimpse into what is happening on the ground, so analytics with such imagery can prove very valuable.  Specifically, the rapid extraction of both vehicles and road networks from overhead imagery allows a host of interesting problems to be tackled, such as congestion mitigation, optimized logistics, evacuation routing, etc.  

A reliable proxy for human population density is critical for effective response to natural disasters and humanitarian crises. A multitude of efforts, such as [SpaceNet](https://spacenet.ai), have striven to improve foundation mapping of building footprints from satellite imagery, which can subsequently be used to population density estimates, pre/post disaster baselines, and numerous other analytics.  Yet while buildings are useful for longer term planning, they rarely provide  information on highly dynamic timescales.  

Automobiles provide an alternative proxy for human population.  Many buildings (eg. office parks or industrial zones) remain unoccupied for large portions of the week.  People tend to stay near their cars, however, so knowledge of where cars are located in real-time provides value in disaster response scenarios.  Simply knowing where vehicles are located is extremely helpful in order to ascertain where resources should be ideally deployed.  In this project, we deploy the [YOLTv5](https://github.com/avanetten/yoltv5) codebase to rapidly identify and geolocate vehicles over large areas.  Geolocations of all the vehicles in an area allow responders to prioritize response areas.

Yet vehicle detections really come into their own when combined with road network data.  We use the [CRESI](https://github.com/avanetten/cresi) framework to extract road networks with travel time estimates, thus permitting optimized routing.  The CRESI codebase is able to extract roads with only imagery, so flooded areas or obstructed roadways will sever the CRESI road graph; this is crucial for post-disaster scenarios where existing road maps may be out of date and the route suggested by navigation services may be impassable or hazardous.  

Diet Hadrade provides a number of graph theory analytics that combine the CRESI road graph with YOLTv5 locations of vehicles.  This allows congestion to be estimated, as well as optimal lines of communication and evacuation scenarios.  In the cells below we provide an example of the analytics that can be performed with this codebase.  Of particular note, the test city selected below (Dar Es Salaam) is not represented in any of the training data for either CRESI or YOLTv5.  The implication is that the Diet Hadrade methodology is quite robust and can be applied immediately to unseen geographies whenever a new need may arise.

See [diet_hadrade.ipynb](https://github.com/avanetten/diet_hadrade/blob/main/notebooks/diet_hadrade.ipynb) for a tutorial, which yields outputs like the images below.

Also see our [explainer video](https://www.youtube.com/watch?v=QUjztPmKFf8) and [blog](https://medium.com/geodesic/the-potent-mix-of-machine-learning-satellite-imagery-vehicles-and-roads-diet-hadrade-34795396c39e).

-----
![Alt text](/figs/cars+roads0.png?raw=true "")
Vehicles (boxes) + roads (yellow/orange lines) extracted in Dar Es Salaam.

-----
![Alt text](/figs/evac0.png?raw=true "")
Optimal evacuation routes for vehicles of interest (trucks in this case).

-----
![Alt text](/figs/crit_nodes0.png?raw=true "")
Critical nodes in the road network for the above evacuation scenario.



