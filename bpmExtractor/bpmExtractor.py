import xml.etree.ElementTree as ET
import datetime
import json

# Parse the XML file
tree = ET.parse("C:/Users/Benja/dev/dtu/iBeat/bpmExtractor/test.xml")
root = tree.getroot()

# Get today's date
today = datetime.date.today()

# Initialize an empty dictionary to store the track info
tracks_info = {}

# Iterate over each TRACK element in the XML
for track in root.iter("TRACK"):
    if track.get("DateAdded") is None:
        continue
    date_added = datetime.datetime.strptime(track.get("DateAdded"), "%Y-%m-%d").date()

    # Check if the track was added today or later
    if date_added >= today:
        # Extract the required attributes
        name = track.get("Name")
        average_bpm = track.get("AverageBpm")
        bit_rate = track.get("BitRate")
        sample_rate = track.get("SampleRate")
        tonality = track.get("Tonality")
        tempos = []
        for tempo in track.iter("TEMPO"):
            this_tempo = {}
            this_tempo["tempo_inizio"] = tempo.get("Inizio")
            this_tempo["tempo_bpm"] = tempo.get("Bpm")
            this_tempo["tempo_metro"] = tempo.get("Metro")
            this_tempo["tempo_battito"] = tempo.get("Battito")
            tempos.append(this_tempo)

        # Store the attributes in the dictionary
        tracks_info[name] = {
            "AverageBpm": average_bpm,
            "BitRate": bit_rate,
            "SampleRate": sample_rate,
            "Tonality": tonality,
            "Tempos": tempos,
        }


# Write the dictionary to a JSON file
with open("tracks_info.json", "w") as f:
    json.dump(tracks_info, f)
