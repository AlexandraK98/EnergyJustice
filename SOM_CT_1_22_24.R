#add in Kohonen via tools > install packages 
#set working directory 
setwd("C:/Users/ack98/OneDrive/Documents/EJ_SOM")


#load libraries 
library(kohonen)
library(ggplot2)
library(rgdal)
library(rgeos)
library(gridExtra)
library(grid)

#read in the shapefile for Houston census tracts 

houston_map <- readOGR(dsn="C:/Users/ack98/OneDrive/Documents/EJ_SOM/hou_CT.shp",layer="hou_CT")

#convert shp object into lat long for better use with ggmap
houston_map <-spTransform(houston_map,CRS("+proj=longlat +datum=NAD83 +no_defs"))

#convert shp object to dataframe
hou_data <- as.data.frame(houston_map)

#convert spatial polygon to dataframe using columns of spatial information
hou_fort <- fortify(houston_map, region = "GEOID")

#merge 2 dataframes 

hou_fort <- merge(hou_fort,hou_data, by.x = "id",by.y="GEOID")

#scale the data 
vars <- hou_data[, c(3,4,5,6,17)]
print(vars)

#standardise between 0 and 1 
scale_vars <- apply(vars, MARGIN = 2, FUN = function(X)(X-min(X))/diff(range(X)))
print(scale_vars)

#now train the SOM with the scaled variables 
#standardise the data creating z-scores and convert to a matrix
data_train_matrix <- as.matrix(scale(scale_vars))
#keep the column names of data_train as names in new matrix 
names(data_train_matrix) <- names(scale_vars)

########## SOM GRID
#define the size, shape and topology of the som grid
som_grid <- somgrid(xdim = 13, ydim=9, topo="hexagonal", neighbourhood.fct="gaussian")

########## TRAIN
# Train the SOM model, alpha is learning rate, rlen is number of iterations
som_model <- som(data_train_matrix, 
                 grid=som_grid, 
                 rlen=500, 
                 alpha=c(0.1,0.01), 
                 keep.data = TRUE )

# Plot of the training progress - how the node distances have stabilised over time.
# mean distance to closes codebook vector during training
plot(som_model, type = "changes")

#load in colors
library(viridis)

###PLOT COUNT
#counts within nodes
plot(som_model, type = "counts", main="Node Counts per Node", palette.name=viridis, shape="straight", border="transparent")



###CREATE PLOTS OF QUALITY AND NEIGHBOUR DISTANCES
par(mfrow = c(1,2)) #create both plots next to each other
#map quality
plot(som_model, type = "quality", main="Distances within Nodes (Quality)", palette.name=grey.colors, shape="straight", border="darkgrey")
#neighbour distances
plot(som_model, type="dist.neighbours", main = "Distances to Neighbouring Nodes", shape="straight", palette.name=grey.colors, border="darkgrey")


dev.off() #mfrow off

#code spread, plot codebook vectors
plot(som_model, main="Codebook Vectors", type = "codes", shape="straight", bgcol="lightgrey", palette.name=rainbow, border="darkgrey")



############################### CLUSTERS
# Form clusters on grid
## use hierarchical clustering to cluster the codebook vectors
som_cluster <- cutree(hclust(dist(getCodes(som_model))), 5)


# Colour palette definition
class_colors <- c("coral3","orange3", "darkseagreen4", "khaki3", "lightpink3")

# plot codes with cluster colours as background
plot(som_model, type="codes", bgcol = class_colors[som_cluster], main = "Clusters", shape="straight", palette.name=rainbow, border="transparent")
add.cluster.boundaries(som_model, som_cluster)


##### MAKE GEOGRAPHIC MAP

#create dataframe of the small area id and of the cluster unit
cluster_details <- data.frame(id=hou_data$GEOID, cluster=som_cluster[som_model$unit.classif])

#we can just merge our cluster details onto the fortified spatial polygon dataframe we created earlier
mappoints <- merge(hou_fort, cluster_details, by.x="id", by.y="id")

# Finally map the areas and colour by cluster
ggplot(data=mappoints, aes(x=long, y=lat, group=group, fill=factor(cluster))) + 
  geom_polygon(colour="transparent")  + 
  coord_equal() + 
  scale_fill_manual(values = class_colors) 


# combine map with cluster details
clus_map <- merge(houston_map, cluster_details, by.x="GEOID", by.y="id")








