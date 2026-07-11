# Posture Study

Stuff that needs to be fixed
- The data logging is only based on alerts
- the detection should end if the session ends
- UI for final caliibration is off screen

Things to add
- feedback in case posture is detected as bad but not as per user --> feedback button
- running script that turns data logged to zip file for easy sharing for user

Stuff to check before particpant starts study
- correct python version
- single file to run and setup entire project in computer - MacOS or Windows
- Video capture happens 
    - permissions for MacOS or Windows
    - browser permissions for MacOS or Windows
- visual prompt if possible which can be dismissed --> guided tour type
- tackling errors that may arise 
    - not enough visbility
- Do not wear earphones to expect to hear the sound --> won't work 

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
- If you are having an annoying day or busy day, don't use the system. If you are bored and want to sit and do work but also have some fun maybe use the system
- might detect a posture that is bad but may not be the same case for you
- System may be waiting for your input at the end of calibration, button may be offscreen

Hyperparameters set for us to consider:
- timeout for considering a postue alert is ignored : no response for 30s (app.py : line 296)
- thresholds for posture (app.py : line 144)
- number of bad posture identified for alert (app.py : line 277)

Interesting thinga to mention
- https://www.reddit.com/r/explainlikeimfive/comments/s1djn2/eli5_why_are_sitting_positions_that_are_bad_for/
- I don't remember where I read this but I recall this reply from a doctor to the question "Which is the best posture?"

Answer: "The next one."
- out of developomen debug=False

Things to consider 
- if the user is told that his posture is to be monitored for correction, the use pruposely tries to not resort to a bad posture
- if the user isnt told the point of the system, then he wont understand the reason behind the beeps if making a bad posture
- the user should know that the system makes beeps, it makes beeps if a bad posture is detected, let it be up to the user to correct their posture
    - then tell the user, prolonged bad posture may cause periodic beeps, correcting the posture may help in this regard
- goal of the system to correct the user ? aware the user ? challenge the user ? 
-peple correct themselves just to not hear the beep ? 
- if we tell the user that the machine can be wrong in predicting the bad posture than some bad posture positions would be considered as false positives by the user and give annoyance
- mention the point of the study is to get the natural seating flow of the person when doing their work, how their posture changes with the work and if this posture can be corrected
