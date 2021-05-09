![image](https://github.com/patternizer/glosat-short-segments/blob/master/short-segment-method-poster-africa.png)
![image](https://github.com/patternizer/glosat-short-segments/blob/master/short-v-long-segment-stations-all.png)
![image](https://github.com/patternizer/glosat-short-segments/blob/master/short-v-long-segment-stations-africa.png)
![image](https://github.com/patternizer/glosat-short-segments/blob/master/sshort-segment-method.png)

# glosat-short-segments

Python code to extract short-segment LSAT absolute temperature timeseries for a selected continent and use a linear model of change in level in 20CRv3 reanalysis to scale the segment to calculate the 1961-1990 baseline normals needed to generate the temperature anomalies. Part of ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `lut-extract-short-segment-stations.py` - python code for extracting stations per continent for the short-segment analysis
* `short-segment-analysis-station.py` - python code to perform short-segment analysis at the station level
* `short-segment-analysis-africa.py` - python code to perform short-segment analysis for all current GloSAT.p03 African stations and compare with the background 20CRv3 ensemble average
* `short-segment-analysis-arctic.py` - python code to perform short-segment analysis for all current GloSAT.p03 Arctic stations and compare with the background 20CRv3 ensemble average


The first step is to clone the latest glosat-short-segments code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-short-segments.git
    $ cd glosat-short-segments

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested in a conda virtual environment running a 64-bit version of Python 3.8+.

glosat-short-segments scripts can be run from sources directly, once the python library dependencies and data input files are satisfied.

Run with:

    $ python lut-extract-short-segment-stations.py
    $ python short-segment-analysis-station.py
    $ python short-segment-analysis-africa.py
    $ python short-segment-analysis-arctic.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

