''''''''''''''''''''''''''''''''''   TheSkyX  '''''''''''''''''''''''''''''

Const AdaptorProcess = "TheSkyXAdaptor.exe"
Const AdaptorPath = "c:\Program Files (x86)\Software Bisque\TheSkyX Professional Edition\Resources\Common\Miscellaneous Files\CustomCOMInterfaces\TheSkyXAdaptor.exe"
If Not IsProcessRunning(AdaptorProcess) Then
    WScript.CreateObject("WScript.Shell").Exec(AdaptorPath)
End If

If Not IsProcessRunning("TheSkyX.exe") Then
  Set TheSkyX_TheSkyObject = CreateObject("TheSkyXAdaptor.RASCOMTheSky")
  Set TheSkyX_ChartObject = CreateObject("TheSkyXAdaptor.StarChart")
  Set TheSkyX_UtilObject = CreateObject("TheSkyXAdaptor.Utils")
  Call TheSkyX_TheSkyObject.Connect()
  wscript.sleep 1000
Else
  wscript.echo "TheSkyX is already running!"
  Set TheSkyX_TheSkyObject = CreateObject("TheSkyXAdaptor.RASCOMTheSky")
  Set TheSkyX_ChartObject = CreateObject("TheSkyXAdaptor.StarChart")
  Set TheSkyX_UtilObject = CreateObject("TheSkyXAdaptor.Utils")
End If


''''''''''''''''''''''''''''''''''   Autoslew  '''''''''''''''''''''''''''''

' start Autoslew
If Not IsProcessRunning("AstroOptikServer.exe") Then
  Set telescope = CreateObject("AstroOptikServer.Telescope")
  telescope.Connected = true
  wscript.sleep 10000
Else
  Set telescope = CreateObject("AstroOptikServer.Telescope")
  wscript.echo "Autoslew is already running!"
End If


'''''''''''''''''''''''''''''''''''   JOB  '''''''''''''''''''''''''''

fileName = "katalog.cat"
'fileName = "zorica1.cat"
'fileName = InputBox("Enter the file name of the object list")

Set fso = CreateObject("Scripting.FileSystemObject")
Set f = fso.OpenTextFile(path & fileName)
br = 0
Do While Not f.AtEndOfStream

	br = br + 1
	
	'read line
	line = f.ReadLine
	wscript.echo br & " : " & line 
Loop
f.Close

Set f = fso.OpenTextFile(path & fileName)
ans = InputBox("Chose your star by numbers on the left")
br = 0
Do While Not f.AtEndOfStream

	br = br + 1
	
	'read line
	line = f.ReadLine
	if br = CDbl(ans) then yourLine = line
	 
Loop
f.Close
wscript.echo
wscript.echo "Chosen star is: " & yourLine
wscript.echo("")

sp = Split(yourLine, " ")
RA_h = sp(1)
Dec_d = sp(2)
wscript.echo "   Object's RA 2000 [h:m:s]: " & RA_h
wscript.echo "   Object's Dec 2000 [d:m:s]: " & Dec_d
'MsgBox Msg
'wscript.quit


'convert into decimal form
sp = Split(RA_h, ":")
RA_h = CDbl(sp(0))+(CDbl(sp(1))+CDbl(sp(2))/60)/60
sp = Split(Dec_d, ":")
if CDbl(InStr(sp(0),"-")) = 0 Then
   Dec_d = CDbl(sp(0))+(CDbl(sp(1))+CDbl(sp(2))/60)/60
ElseIf CDbl(InStr(sp(0),"-")) > 0 Then
   Dec_d = -1*(Abs(CDbl(sp(0)))+(CDbl(sp(1))+CDbl(sp(2))/60)/60)
End if
wscript.echo("")
wscript.echo "   Object's RA 2000 [hours]: " & RA_h
wscript.echo "   Object's Dec 2000 [degrees]: " & Dec_d


result = TheSkyX_UtilObject.Precess2000ToNow(RA_h, Dec_d)
RA_h = result(0)
Dec_d = result(1)
wscript.echo("")
wscript.echo "   Object's RA Now [hours]: " & RA_h
wscript.echo "   Object's Dec Now [degrees]: " & Dec_d

' slew
telescope.SlewToCoordinates RA_h, Dec_d
'wscript.sleep(1000000)


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
