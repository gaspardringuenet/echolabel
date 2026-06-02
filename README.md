# `echolabel` - Minimalist Interactive Echogram Data Extraction Tools

![](image.png)

## The [Echoregions](https://echoregions.readthedocs.io/en/latest/index.html)-based approach

This branch is dedicated to refactoring the first version of `echolabel` (currently the 'main' branch). The goal is to switch from a monolithic / standalone app to increase modularity through `echostack` dependencies. Refactoring will first primarily focus on the *labelling* process (drawing polygons or "shapes" on echograms and fetching underlying data). The *extraction* process (performing segmentation tasks on shapes using an interactive Dash app) will be refactored later on.

### Core functionalities

The purpose of the program is interactive echogram labelling. The desired process can be summarized as:

    Acoustic data -- <GUI> -- Polygons / Lines coordinates

The main design choice of `echolabel` is to wrap [Labelme](https://labelme.io/) (an image annotation software). This design leverages an powerful app with greate UX, and requires little computing power as the data is reduced to simple images during annotation. This enables an easy and fast annotaton workflow (with the major limitation of losing real time, depth and position labels for the data and enforcing a 1:1 width ratio for pixels).

In order to achieve this, we chose to implement the following functinalities:

- a CLI
- an acoustic data loader
- an image dataset manager, in charge of printing images out of the MVBS data to use as input for Labelme
- a Labelme wrapper, in charge of parsing Labelme's output (json files) and syncing them with a usable representation of the shapes

### Limitations of the previous attempt

- **Data loading and compatibility** - Loading was tailored to our own data files (IRD NetCDF data following the [IMOS SOOP-BA convention](https://imos.org.au/wp-content/uploads/2024/06/SOOP-BA_NetCDF_Conventions_Version_2.2.pdf)) and variable names are hardcoded. The risk of failure with other data formats and convention is quite high.
- **Weak annotation coordinates parsing & dependency to image names** - Labelme produces annotations in the form of a list of points in each image's (x, y) coordinate system. This cannot be used directly to index the MVBS data. The issue has circumvented by encoding the *offset* (index of the start in the parent raw array) of each image into its name. Other infos were encoded in the name, which is poorly maintainable.
- **Internal SQLite database** - The hierarchical nature of the data (MVBS > Images ; MVBS > Annotations > Echotypes) was enforced using an SQLite database. This approach presented several benefits: clear links between data objects; easy iteration ; cascading deletion ; relational operations ; easy use in a Dash app for echotypes extraction. However, it required a large amount of code, was hard to maintain (each choice of attributes in a table can turn out to be wrong, and is hardcoded in several files), and eventually destroyed the projects modularity: all softwares (images manager, annotation, echotypes extraction) relying on the same database, which is hidden from the user.
- **Incomplete management of produced images datasets** - The images datasets manager is incomplete and needs to be both simplified (remove circumvoluted images names, database reliance) and extended (clear definition of when a images datasets is created or deleted from cache)

### Design changes

Here we propose to :

- **Use [Echopype](https://echopype.readthedocs.io/en/latest/)'s format as data loading convention**, and limit dependency to the loading (by converting variable / coordinates to generics a single time). This will increase compatibility by leveraging a widely used format & increase maintainability (extending compatibilty should be easy aftwerward).
- **Ditch the SQL database** in favor of more meaningful data files. This is the main modification to the first version. By parsing Labelme's annotations into Echoregions objects, we will be able to produce much more useful and interoperable echoregions files (e.g.,  .csv or .evr for 2D regions). Enforcing data consistency (e.g., hierarchy, uniqueness) will be performed using a combination of output files, manifest files (e.g. "which raw data was used for this output?", "what are the coordinates of this image in the raw MVBS dataset?") and python checks.
- **Clearly separate concerns between cache and output directories**
- **Handle cache explicitly**

## Challenges and proposed solutions

### Main challenges

- Labelme - Echoregions synchronization
- Uniqueness
- Cache: Need to figure out to avoid storing dozen of images datasets (a new one is created whenever viz-parmas change). A possible routine would be
  - overwriting an existing visualization (not a real loss since `echolabel` is by design not very suited to frequently switching between visualizations anyway), thus ensuring no memory usage explosion.
  - clearing cache with a --clear-cache command if necessary.

### Other

- Labelme config - edit the Labelme config file to modify part of the GUI (e.g., remove functionalities which are useless / cannot be translated reliably into Echoregions)
- Useful modifications to Labelme - nearest-neighbour image interpolatio (would require forking)