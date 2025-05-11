''''''''''''
'1. connect telescope; NO NEED FOR THIS IF YOU DO NOT SLEW THE TELECOPE TO WEST!!!
'2. connect camera
'3. using TheSky calculate when is the sunset and wait it; while waiting the telescope's tracking is forced to false
'4. after sunset start the loop
'LOOOOP
'	a) measure the average level in the central 50 pixels
'	b) if the level drops bellow 45000 start shooting in the B band with exposure time 5 secs
'	c) stop shooting after 5 exposures
'	d) change to V band and repeat a, b, c
'	e) change to R band and repeat a, b, c
'	f) change to I band and repeat a, b, c
'       g) telescope.Tracking = false for any case
'NEXT
'5. take 10 bias frames
'6. take 10 dark frames with 5 sec
'7. take 5 dark frames with 180 sec for later if needed for scalable-dark calibration
'8. make master frames

'NOTE:
'	v2 version does dithering
'	v4 is a step toward more universal script which would work on 60cm and 1.4m telescope (worked fine on 1.4m telescope on 08.12.2016)
'	v8 civil and nautical twilight added


'''''''''''''''''''''''''''''''' SET CONSTANTS AND SO ON ''''''''''''''''''''''''''''''


Set fso = CreateObject("Scripting.FileSystemObject")
pathFlName = fso.GetFile("takeFlats_universal_v8.vbs").Path
path = fso.GetParentFolderName(pathFlName) & "\"



' First constants with questions
ifUpdateTime = InputBox("Do you want to UPDATE the time [y/n]")
ifBinning2x2 = InputBox("Do you want 2x2 BINNING [y/n]")
ifDithering = InputBox("Do you want DITHERING [y/n]")
sunsetORsunrise = InputBox("When do you take flats [sunset/sunrise]")
filterPosition = InputBox("Enter filter position in the wheel which to be used separated by comma [0,1,2,3,4]")
filterLabel = InputBox("Enter the corresponding filter label [B,V,R,I,L]")
CCDtype = InputBox("Enter the CCD type [U42,ST10,IKONL]")
TELESCOPEtype = InputBox("Enter the TELESCOPE type [EQ60,AZ140]")




' Second fixed constants
Const startShift_ATsunset = +20   'in minutes (minust is for BEFORE; plus if for AFTER)
Const startShift_ATsunrise = -60   'in minutes (minust is for BEFORE; plus if for AFTER)
Const twilightType = "none"    ' options are: none (0 degree), civil (6 degree), nautical (12 degree); difference in time between each is ~50min
Const numBiases = 10  		' recommended to take 10
Const numDarks = 10			' recommended to take 10
Const numScDarks = 10	' recommended to take 10
Const expTime_forScDark = 200	' exposure time for scalable dark procedure; recommended to take 5 * the longest for the image
Const size = 2048    'Define the size of the subframe
Const ifVerbose = "y"
Const ifCheck = "n"



'Third, conditional constants
if sunsetORsunrise = "sunrise" AND twilightType = "none" Then
	propNum = 67
End If
if sunsetORsunrise = "sunrise" AND twilightType = "civil" Then
	propNum = 167
End If
if sunsetORsunrise = "sunrise" AND twilightType = "nautical" Then
	propNum = 169
End If
if sunsetORsunrise = "sunset" AND twilightType = "none" Then
	propNum = 69
End If
if sunsetORsunrise = "sunset" AND twilightType = "civil" Then
	propNum = 168
End If
if sunsetORsunrise = "sunset" AND twilightType = "nautical" Then
	propNum = 170
End If



' CCD params
if CCDtype = "U42" Then
	numFlats = 5			' recommended to take X
	eksp_forFlat = 5
	If sunsetORsunrise = "sunset" Then
		ADU_l = 5000  'on 2012_06_08 the script couldn't finish all 5 flats in I band because this limit was 20000
		ADU_u = 47000
	else
		ADU_l = 10000
		ADU_u = 47000
	End If
End If
if CCDtype = "ST10" Then
	numFlats = 8			' recommended to take X
	eksp_forFlat = 1
	If sunsetORsunrise = "sunset" Then
		ADU_l = 5000  'on 2012_06_08 the script couldn't finish all 5 flats in I band because this limit was 20000
		ADU_u = 47000
	else
		ADU_l = 10000
		ADU_u = 47000
	End If
End If
if CCDtype = "IKONL" Then
	numFlats = 8			' recommended to take X
	eksp_forFlat = 5
	If sunsetORsunrise = "sunset" Then
		ADU_l = 1000  
		ADU_u = 57000
	else
		ADU_l = 10000
		ADU_u = 66000
	End If
End If



' randomize number
max=50 	'arcsec, that is, ~100 pixels
min=1
Dim arr_RA(100)
Dim arr_DEC(100)
For j = 1 To 100
	'Randomize
	'in decimal hours
	rnd_RA = Int((max-min+1)*Rnd+min)
	rnd_DEC = Int((max-min+1)*Rnd+min)
	if rnd_RA < max/2 then
	   arr_RA(j) = (rnd_RA/60/60)*(1/15)
	else
	   arr_RA(j) = -1*(rnd_RA/60/60)*(1/15)
	end if
	if rnd_DEC < max/2 then
	   arr_DEC(j) = rnd_DEC/60/60
	else
	   arr_DEC(j) = -1*(rnd_DEC/60/60)
	end if
	'wscript.echo rnd_RA
	'wscript.echo arr_RA(j)
	'wscript.echo arr_DEC(j)
	'wscript.echo "******"
Next
'MsgBox Msg
'wscript.quit


''''''''''''''''''''''' FILTER-WHEEL ISSUES ''''''''''''''''''''''

filterPosition_array = Split(filterPosition, ",")
num_filterPosition = Ubound(filterPosition_array) + 1

filterLabel_array = Split(filterLabel, ",")
num_filterLabel = Ubound(filterLabel_array) + 1

' quick safety check
if num_filterPosition <> num_filterLabel Then
	MsgBox "The filter positions and filter numbers do not match in number! Please START AGAIN"
	wscript.quit
End If


'''''''''''''''''''''''''''''''   THESKY ISSUES   ''''''''''''''''''''''''


' determine the THESKY version
Set folderTHESKY = fso.GetFolder("c:\Program Files (x86)\Software Bisque\")
Set fc = folderTHESKY.SubFolders
For Each f1 in fc
        s = s & f1.name
        s = s & vbCrLf
Next

ifExist = InStr(s,"TheSkyX")
if ifExist = 1 Then
	wscript.echo "TheSky version is: TheSkyX"
	wscript.echo ""

	' define constants
	Const AdaptorProcess = "TheSkyXAdaptor.exe"
	Const AdaptorPath = "c:\Program Files (x86)\Software Bisque\TheSkyX Professional Edition\Resources\Common\Miscellaneous Files\CustomCOMInterfaces\TheSkyXAdaptor.exe"

	' define variables
	If Not IsProcessRunning(AdaptorProcess) Then
	    WScript.CreateObject("WScript.Shell").Exec(AdaptorPath)
	End If

	If Not IsProcessRunning("TheSkyX.exe") Then
	  Set TheSky_TheSkyObject = CreateObject("TheSkyXAdaptor.RASCOMTheSky")
	  Set TheSky_ChartObject = CreateObject("TheSkyXAdaptor.StarChart")
	  Set TheSky_UtilObject = CreateObject("TheSkyXAdaptor.Utils")
	  Call TheSky_TheSkyObject.Connect()
	  wscript.sleep 1000
	Else
	  wscript.echo "TheSkyX is already running!"
	  Set TheSky_TheSkyObject = CreateObject("TheSkyXAdaptor.RASCOMTheSky")
	  Set TheSky_ChartObject = CreateObject("TheSkyXAdaptor.StarChart")
	  Set TheSky_UtilObject = CreateObject("TheSkyXAdaptor.Utils")
	End If

Else
	ifExist = InStr(s,"TheSky6")
	if ifExist = 1 Then
		wscript.echo "TheSky version is: TheSky6"
		wscript.echo ""

		' define variables
		Set TheSky_TheSkyObject = CreateObject("TheSky6.RASCOMTheSky")
		Set TheSky_ChartObject = CreateObject("TheSky6.StarChart")
		Set TheSky_UtilObject = CreateObject("TheSky6.Utils")
		Call TheSky_TheSkyObject.Connect()

	Else
		wscript.echo "TheSky6 and TheSkyX does NOT exist"
    End If
End If



Dim objShell

'''''''''''''''''''''''''''''''''' 	UPDATE THE TIME  '''''''''''''''''''''''''''''''''

if ifUpdateTime = "y" then

	Set objShell = Wscript.CreateObject("WScript.Shell")

	objShell.Run chr(34) & "c:\Program Files\D4\D4.exe", 1, false
	Set objShell = Nothing

	'MsgBox Msg
	'wscript.quit

end if



''''''''''''''''''''''''''''''''' telescope issues '''''''''''''''''''''''''''''''''''''

'connect telescope to make sure that the tracking is off
Set telescope = CreateObject("AstroOptikServer.Telescope")
telescope.Connected = true
if not telescope.Connected then
      wscript.echo "Telescope Link Failed"
      'ALARM
end if
wscript.sleep 5000
telescope.Tracking = false



''''''''''''''''''''''''''''''''' camera issues '''''''''''''''''''''''''''''''''''''

' Set up the camera
Dim cam ' "The" Camera object
Set cam = CreateObject("MaxIm.CCDCamera")
cam.LinkEnabled = True
cam.DisableAutoShutdown = True			' Leave camera on when we are done
if Not cam.LinkEnabled Then
   wscript.echo "Failed to start camera."
   Quit
End If


'cool down the camera if needed
camStatus = InputBox("Is the camera cooled down [y/n]")
If camStatus <> "y" Then
	' Cool down the camera
	cam.CoolerOn = true
	cam.TemperatureSetPoint = InputBox("Enter Temperature Point")
	Do While Not cam.Temperature = cam.TemperatureSetPoint
	   wscript.Sleep 100
	Loop
End If

'bin
If ifBinning2x2 = "y" Then
	cam.BinX = 2
	cam.BinY = 2
End If



''''''''''''''''''''''''''''''''' sleep '''''''''''''''''''''''''''''''''''''



' common features

'winter or summer time
summORwin = InputBox("Summer or winter civil time [s/w]")

'set location
dJD = 0    'ignored if UseCompterClock=1
nDST = 0     'see c:\Documents and Settings\ovince\My Documents\Software Bisque\TheSkyX\Locations.txt
bUseComputerClock = 1    'to use the computer's time; 1=Yes 0=No
szLoc = "Vidojevica"   'Location name
dLong = -21.555666     'Longiude
dLat = 43.1401666     'LAtitude
dTZ = 1	 'Time Zone
dElev = 1133     'elevation in meters
Call TheSky_TheSkyObject.SetWhenWhere(dJD, nDST, bUseComputerClock, szLoc, dLong, dLat, dTZ, dElev)
Call TheSky_ChartObject.Refresh
wscript.sleep 2000

'current local time
CurTime = Now()
currLT = FormatDateTime(CurTime, vbShortTime)
wscript.echo "Current Local Time: " & currLT
wscript.echo ""

' convert time into sleep mode i.e. decimal form
currLT = Split(currLT,":")(0) + Split(currLT,":")(1)/60




If sunsetORsunrise = "sunset" Then

	'position of the sun
	Set InfoSun = TheSky_ChartObject.Find("Sun")
	sunSetTime_LT = InfoSun.Property(propNum)
	if summORwin = "s" then sunSetTime_LT = sunSetTime_LT + 1.
	if summORwin = "w" then sunSetTime_LT = sunSetTime_LT + 0.
	
	'wscript.echo InfoSun.Property(69)
	'wscript.echo InfoSun.Property(168)
	'wscript.echo InfoSun.Property(170)
	'wscript.echo InfoSun.Property(propNum)
	'MsgBox "ENTER"

	sunSetTime_LT_h = Int(sunSetTime_LT)
	sunSetTime_LT_m = Int((sunSetTime_LT - sunSetTime_LT_h)*60)
	sunSetTime_LT_s = Int((((sunSetTime_LT - sunSetTime_LT_h)*60)-sunSetTime_LT_m)*60)
	sunSetTime_LT_all = "(" & sunSetTime_LT_h & ":" & sunSetTime_LT_m & ":" & sunSetTime_LT_s & ")"
	wscript.echo "SUN sets at: " & Round(sunSetTime_LT,5) & " " & sunSetTime_LT_all & " hours Local Time"
	wscript.echo ""

	if currLT < sunSetTime_LT then
	   sleepTime = (sunSetTime_LT - currLT)+startShift_ATsunset/60. ' startShift [min] converted to [hours]
	   wscript.echo "Time Left To Sunset: " & Round((sunSetTime_LT - currLT),2) & " [hours]"
	   wscript.echo "Time Left To Observation: " & Round(sleepTime,2) & " [hours]"
	   wscript.echo
	   currSleepTime = 0
	   Do While currSleepTime < sleepTime
		wscript.Sleep 10000 '10 secunds
		currSleepTime = currSleepTime + 10/60/60
		timeLeftToObserve = sleepTime - currSleepTime
		wscript.echo "Time Left To Observation: " & Round(timeLeftToObserve,2) & " [hours]  OR " & Round(timeLeftToObserve*60,2) & " [minutes]"
		telescope.Tracking = false
	   Loop
	else
	   wscript.echo "The Sunset Already Past; let's try"
	end if
End If




If sunsetORsunrise = "sunrise" Then

	'position of the sun
	Set InfoSun = TheSky_ChartObject.Find("Sun")
	sunRiseTime_LT = InfoSun.Property(propNum)
	if summORwin = "s" then sunRiseTime_LT = sunRiseTime_LT + 1.
	if summORwin = "w" then sunRiseTime_LT = sunRiseTime_LT + 0.
	
	'wscript.echo InfoSun.Property(67)
	'wscript.echo InfoSun.Property(167)
	'wscript.echo InfoSun.Property(169)
	'wscript.echo InfoSun.Property(propNum)
	'MsgBox "ENTER"

	sunRiseTime_LT_h = Int(sunRiseTime_LT)
	sunRiseTime_LT_m = Int((sunRiseTime_LT - sunRiseTime_LT_h)*60)
	sunRiseTime_LT_s = Int((((sunRiseTime_LT - sunRiseTime_LT_h)*60)-sunRiseTime_LT_m)*60)
	sunRiseTime_LT_all = "(" & sunRiseTime_LT_h & ":" & sunRiseTime_LT_m & ":" & sunRiseTime_LT_s & ")"
	wscript.echo "SUN rises at: " & Round(sunRiseTime_LT,5) & " " & sunRiseTime_LT_all & " hours Local Time"
	wscript.echo ""

	if currLT < sunRiseTime_LT then
	   sleepTime = (sunRiseTime_LT - currLT)+startShift_ATsunrise/60. '1h before sunrise
	   wscript.echo "Time Left To Sunrise: " & Round((sunRiseTime_LT  - currLT),2) & " [hours]"
	   wscript.echo "Time Left To Observation: " & Round(sleepTime,2) & " [hours]"
	   wscript.echo
	   currSleepTime = 0
	   Do While currSleepTime < sleepTime
		wscript.Sleep 10000 '10 secunds
		currSleepTime = currSleepTime + 10/60/60
		timeLeftToObserve = sleepTime - currSleepTime
		wscript.echo "Time Left To Observation: " & Round(timeLeftToObserve,2) & " [hours]  OR " & Round(timeLeftToObserve*60,2) & " [minutes]"
		telescope.Tracking = false
	   Loop
	else
	   wscript.echo "The Sunrise Already Past; let's try"
	end if
End if

'MsgBox "ENTER"
'wscript.quit



''''''''''''''''''''''''''''''''' LOOOOOP '''''''''''''''''''''''''''''''''''''


' Set up the location and size of the subframe
Dim left, top
If ifBinning2x2 = "y" Then
	left = (cam.CameraXSize/2 - size)/2
	top = (cam.CameraYSize/2 - size)/2
	'wscript.echo "left: " & left
	'wscript.echo "top: " & top
Else
	left = (cam.CameraXSize - size)/2
	top = (cam.CameraYSize - size)/2
	'wscript.echo "left: " & left
	'wscript.echo "top: " & top
End If


N_dith = 1
If sunsetORsunrise = "sunset" Then

	brojac = 1
	i = 0
	For N = 1 to num_filterPosition

		cam.NumX = size
		cam.NumY = size
		cam.StartX = left
		cam.StartY = top

		filterNumber = filterPosition_array(i)
		suff = filterLabel_array(i)
		i = i + 1

		wscript.echo
		wscript.echo "FILTER: " & suff

		Do
			result = ADU(eksp_forFlat, filterNumber)
			wscript.echo "Level: " & result

			' Save if ADU is satisfied
			if result < ADU_u AND result > ADU_l Then


				if brojac = 1 then

					'if dithering needed => make tracking TRUE
					if ifDithering = "y" AND N = 1 Then
					   telescope.Tracking = true
					   wscript.sleep 3000  ' wait 3 seconds
					   RA_h_ref = telescope.RightAscension
   					   DEC_d_ref = telescope.Declination
					end if

					'set full frame size
					cam.SetFullFrame
					result = ADU(eksp_forFlat, filterNumber)

				end if

				'if dithering needed => do dithering
				if ifDithering = "y" Then

				   ' Add random
				   RA_h = RA_h_ref + arr_RA(N_dith)
				   DEC_d = DEC_d_ref + arr_DEC(N_dith)
				   wscript.echo "diff_RA: " & RA_h_ref-RA_h & " hours"
				   wscript.echo "diff_DEC: " & DEC_d_ref-DEC_d & " degree"
				   N_dith = N_dith + 1

				   'slew the telescope and wait a bit
				   telescope.SlewToCoordinates RA_h, DEC_d
				   wscript.sleep 3000  ' 3 seconds
				end if

				wscript.echo "   Image Number: " & brojac
				cam.SetFITSKey "IMAGETYP","FLAT"
				cam.SaveImage path & "flatZalaz-" & Cstr(eksp_forFlat) & "sec-" & brojac & "_" & suff & ".fit"
				brojac = brojac + 1


			End If

		Loop While result > ADU_l AND brojac <= numFlats

		brojac = 1

	Next
End If


If sunsetORsunrise = "sunrise" Then

	brojac = 1
	i = 0
	For N = 1 to num_filterPosition

		cam.NumX = size
		cam.NumY = size
		cam.StartX = left
		cam.StartY = top

		filterNumber = filterPosition_array(i)
		suff = filterLabel_array(i)
		i = i + 1

		wscript.echo
		wscript.echo "FILTER: " & suff

		Do
			result = ADU(eksp_forFlat, filterNumber)
			wscript.echo "Level: " & result

			' Save if ADU is satisfied
			if result < ADU_u AND result > ADU_l Then


				if brojac = 1 then

					'if dithering needed => make tracking TRUE
					if ifDithering = "y" AND N = 1 Then
					   telescope.Tracking = true
					   wscript.sleep 3000  ' wait 3 seconds
					   RA_h_ref = telescope.RightAscension
   					   DEC_d_ref = telescope.Declination
					end if

					'set full frame size
					cam.SetFullFrame
					result = ADU(eksp_forFlat, filterNumber)

				end if


				'if dithering needed => do dithering
				if ifDithering = "y" Then

				   ' Add random
				   RA_h = RA_h_ref + arr_RA(N_dith)
				   DEC_d = DEC_d_ref + arr_DEC(N_dith)
				   wscript.echo "diff_RA: " & RA_h_ref-RA_h & " hours"
				   wscript.echo "diff_DEC: " & DEC_d_ref-DEC_d & " degree"
				   N_dith = N_dith + 1

				   'slew the telescope and wait a bit
				   telescope.SlewToCoordinates RA_h, DEC_d
				   wscript.sleep 3000  ' 3 seconds
				end if


				wscript.echo "   Image Number: " & brojac
				cam.SetFITSKey "IMAGETYP","FLAT"
				cam.SaveImage path & "flatIzlaz-" & Cstr(eksp_forFlat) & "sec-" & brojac & "_" & suff & ".fit"
				brojac = brojac + 1
			End If
		Loop While result < ADU_u AND brojac <= numFlats

		brojac = 1

	Next

End if


'''''''''''''''''''''''''''''    CLOSE TELESCOPE  ''''''''''''''''''''''''''

'set full frame for bias and dark frames
cam.SetFullFrame

' close the cover if sunrise since dark frames are bad due to light licking between shutter blades
If TELESCOPEtype = "AZ140" Then
	If sunsetORsunrise = "sunrise" Then
		telescope.closecover
		telescope.park
	End if
	telescope.Tracking = false
	telescope.Connected = false
End If
If TELESCOPEtype = "EQ60" Then
	telescope.Tracking = false
	telescope.Connected = false
End If
If TELESCOPEtype = "AZ140" Then
	If sunsetORsunrise = "sunset" Then
		telescope.closecover
		telescope.park
	End if
	telescope.Tracking = false
	telescope.Connected = false
	End If
If TELESCOPEtype = "EQ60" Then
	telescope.Tracking = false
	telescope.Connected = false
End If

If sunsetORsunrise = "sunrise" Then
	MsgBox "Close the pavillion first and thereafter press OK to continue by taking calibration images !!!"
End If

If sunsetORsunrise = "sunset" Then
	MsgBox "Close the pavillion first and thereafter press OK to continue by taking calibration images !!!"
End If



''''''''''''''''''''''''''''  take Bias frames  ''''''''''''''

'close all frames
Set mxapp = CreateObject("Maxim.Application")
Call mxapp.CloseAll()


'make Bias frames
wscript.echo
For N = 0 to numBiases-1
	wscript.echo "taking " & Cstr(numBiases) & " bias frames"
	wscript.echo
      cam.Expose 0, 0
      Do While Not cam.ImageReady
	' Don't consume CPU time while waiting
	wscript.sleep 100
      Loop
      If sunsetORsunrise = "sunset" Then
		cam.SaveImage path & "biasZalaz-" & Cstr(N+1) & ".fit"
      Else
		cam.SaveImage path & "biasIzlaz-" & Cstr(N+1) & ".fit"
      End if
Next


''''''''''''''''''''''''''''  take Dark frames for flat ''''''''''''''


'make Dark frames
For N = 0 to numDarks-1
	wscript.echo "taking " & Cstr(numDarks) & " dark frames with " & Cstr(eksp_forFlat) & " sec"
	wscript.echo
      cam.Expose eksp_forFlat, 0
      Do While Not cam.ImageReady
	' Don't consume CPU time while waiting
	wscript.sleep 100
      Loop
      If sunsetORsunrise = "sunset" Then
      	cam.SaveImage path & "darkZalaz-" & Cstr(eksp_forFlat) & "sec-" & Cstr(N+1) & ".fit"
      Else
      	cam.SaveImage path & "darkIzlaz-" & Cstr(eksp_forFlat) & "sec-" & Cstr(N+1) & ".fit"
      End if

Next



''''''''''''''''''''''''''''  make Dark frames  for scalable dark later on''''''''''''''


'make Dark frames
For N = 0 to numScDarks-1
	wscript.echo "taking " & Cstr(numscDarks) & " dark frames with " & Cstr(expTime_forScDark) & " sec for scalable dark"
	wscript.echo
      cam.Expose expTime_forScDark, 0
      Do While Not cam.ImageReady
	' Don't consume CPU time while waiting
	wscript.sleep 100
      Loop
      If sunsetORsunrise = "sunset" Then
      	cam.SaveImage path & "scDarkZalaz-" & Cstr(expTime_forScDark) & "sec-" & Cstr(N+1) & ".fit"
      Else
      	cam.SaveImage path & "scDarkIzlaz-" & Cstr(expTime_forScDark) & "sec-" & Cstr(N+1) & ".fit"
      End if
Next


''''''''''''''''''''''''''''  make Dark frames with 600 sec for scalable dark later on''''''''''''''


'make Dark frames with 600 sec
If sunsetORsunrise = "sunrise" Then
	For N = 0 to 9
	      wscript.echo "taking 10 dark frames with 600  for scalable dark"
	      wscript.echo
	      cam.Expose 600, 0
	      Do While Not cam.ImageReady
		' Don't consume CPU time while waiting
		wscript.sleep 100
	      Loop
	      'dark600s-0001_D.fit
	      cam.SaveImage path & "dark600s-" & Right("0000" & N+1, 4) & "_D.fit"
	Next
End If


'''''''''''''''''''''''''''  SEND ALL TO CLOUD AFTER SUNRISE FLATS ''''''''''''''''''''''''''

If sunsetORsunrise = "sunrise" Then

	If TELESCOPEtype = "AZ140" Then

		Set objShell = Wscript.CreateObject("WScript.Shell")

		objShell.Run chr(34) & "%SYSTEMROOT%\System32\wscript.exe" & chr(34) & " " & chr(34) & "d:\Posmatranja\rar-ftp.js" & chr(34), 1, false
		Set objShell = Nothing

	End If

	If TELESCOPEtype = "EQ60" Then

		Set objShell = Wscript.CreateObject("WScript.Shell")

		objShell.Run chr(34) & "%SYSTEMROOT%\System32\wscript.exe" & chr(34) & " " & chr(34) & "d:\Vidojevica60cm\rar-ftp.js" & chr(34), 1, false
		Set objShell = Nothing

	End If

End If



'****************************************************************** FUNCTIONS **********************************

Function ADU(time, filterNumber)

	cam.Expose time, 1, filterNumber 'exposure time, shutter open, filter
	Do While Not cam.ImageReady
		' Don't consume CPU time while waiting
		wscript.sleep 100
	Loop


	' Find average value
	Dim doc
	Set doc = cam.Document
	' CalcAreaInfo returns variant array whose second element is the average
	ADU = doc.CalcAreaInfo(0, 0, doc.XSize-1, doc.YSize-1)(2)
	'wscript.echo "doc.XSize-1: " & doc.XSize-1
	'wscript.echo "doc.YSize-1: " & doc.YSize-1
End Function



'********************************************************************

Function IsProcessRunning(strProcess)
    Dim Process, strObject
    IsProcessRunning = False
    strObject = "winmgmts://"
    For Each Process in GetObject(strObject).InstancesOf("win32_process")
    If UCase(Process.name) = UCase(strProcess) Then
        IsProcessRunning = True
        Exit Function
    End If
    Next
End Function
