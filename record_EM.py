from sksurgerynditracker.nditracker import NDITracker
SETTINGS = {
    "tracker type": "aurora"
        }
TRACKER = NDITracker(SETTINGS)

TRACKER.start_tracking()
port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()



for t in tracking:
  print (t)
TRACKER.stop_tracking()
TRACKER.close()