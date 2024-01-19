

# iBeat

_By Benjamin Banks(s234802), David Svane(234848) og Lucas Pedersen(s234842)_

This was made for 3-weeks project for a course a DTU September 2023. The following is frow our report:

## Abstract
This paper addresses the challenge of beat detection in music, a crucial first step towards live synchronization of visual elements to musical tracks. Previous beat detection methods based on expert systems are either too simple to account for the varieties in music or take a significant amount of development time and expertise. Utilizing that convolutional neural networks are good at finding their own patterns in data, we developed and assessed two convolutional neural network (CNN) systems for identifying beats in 10-second music snippets: one based on waveform (WM) analysis and the other on spectrogram (SM) analysis. Our evaluation shows that the spectrogram-based CNN significantly outperforms the waveform-based approach in beat placement accuracy on a 5% significance level. Our findings might suggest that spectrograms highlight certain features that are helpful for beat tracking, which might be used for further development in the area.

## Results
11 random sample from best to worst F-measure:





<table>
    <tr>
        <td>
            <h2>Waveform</h2>
        </td>
        <td>
            <h2>Spectrogram</h2>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="./audio/wav_1994-967-127000.wav"></source>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_1994-967-127000.wav"></source>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_2003-2109-192000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_2003-2109-192000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_1999-1535-147000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_1999-1535-147000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_2001-1887-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_2001-1887-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_2011-3143-166000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_2011-3143-166000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_2002-1946-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_2002-1946-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_2015-3701-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_2015-3701-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_1996-1166-126000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_1996-1166-126000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_1996-1222-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_1996-1222-0.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_2010-3030-103000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_2010-3030-103000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
    <tr>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/wav_2018-4033-66000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
        <td>
            <audio controls="controls">
                <source type="audio/wav" src="audio/spect_2018-4033-66000.wav"></source>
                <p>Your browser does not support the audio element.</p>
            </audio>
        </td>
    </tr>
</table>