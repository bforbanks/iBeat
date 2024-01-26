

# iBeat

_By Benjamin Banks(s234802), David Svane(234848) og Lucas Pedersen(s234842)_

This was made for 3-weeks project for a course a DTU September 2023. 

## Data and model weights
As much of our data and our model weight files for the models compared in our report were above the github limit of 25MB, it isn't available on git, however, if it is needed for any purpose, reach out to s234842@dtu.dk.


The following is frow our report:

## Abstract
This paper addresses the challenge of beat detection in music, a crucial first step towards live synchronization of visual elements to musical tracks. Previous beat detection methods based on expert systems are either too simple to account for the varieties in music or take a significant amount of development time and expertise. Utilizing that convolutional neural networks are good at finding their own patterns in data, we developed and assessed two convolutional neural network (CNN) systems for identifying beats in 10-second music snippets: one based on waveform (WM) analysis and the other on spectrogram (SM) analysis. Our evaluation shows that the spectrogram-based CNN significantly outperforms the waveform-based approach in beat placement accuracy on a 5% significance level. Our findings might suggest that spectrograms highlight certain features that are helpful for beat tracking, which might be used for further development in the area.

## Results
[Listen to the snippets here](https://www.benjaminbanks.com/dtu/iBeat)

Best snippets above, and 11 random sample from best to worst F-measure underneath.
