# Posture Study

Stuff that needs to be fixed
- Errors in analyze_results.py
- The data logging is only based on alerts
- there is no adaptive feedback mechanism developed
- the detection should end if the session ends

Things to add
- PyAudio : test chime button
- testing for the system lighting is fine, if it is able to detect posture errors
- more postures to be incorporated
- feedback in case posture is detected as bad but not as per user --> feedback button
- running script that turns data logged to zip file for easy sharing for user
- run baseline script that understands the 'normal' seating position of the user
- chime if noo posture identified for long time

Stuff to check before particpant starts study
- correct python version
- single file to run and setup entire project in computer - MacOS or Windows
- Video capture happens 
    - permissions for MacOS or Windows
    - browser permissions for MacOS or Windows
- test a bad posture notification once
- visual prompt if possible which can be dismissed --> guided tour type
- tackling errors that may arise 
    - not enough visbility

Questions to discuss
- tell the user about the study ? --> split cohorts that know and don't know 
- activities other than working on computer ? - talking on phone, adding a detection if person is present or not 

Things to mention to user in writeup document
- do some work for 30 minutes on laptop with this running, connect charger and sit
- close tabs if possible allowing for less lag 
- make sure sound is turned on, test the chime and stay alert for it, tell them the different chimes, no posture and alert for posture correctness
- inform the user about the postures being studied --> its fine if they do not resort to them
- sharing what data files --> after 30 minutes special chime plays --> simply zip files and share
- they should try to sit straight for 30 minutes 
- activities other than working on computer ? - talking on phone 
- data privacy statement --> no video or sound data is recorded --> simply posture data and frequency of that posture is recorded
- tell them not to review the code, cause then they know the posture I am detecting or exactly what the system is doing
- get person focued in their work, becuase we want genuine posture problems to be identified
- play songs while working to distract yourself ?
- upper body above chest is also fine
- check if lighting is fine
- run for baseline script of normal seating position
- chime if no posture identified for long time
- sometimes the software fails on first run, the web cam doesnt work, requires software start again

Hyperparameters set for us to consider:
- timeout for considering a postue alert is ignored : no response for 30s (app.py : line 296)
- thresholds for posture (app.py : line 144)
- number of bad posture identified for alert (app.py : line 277)
