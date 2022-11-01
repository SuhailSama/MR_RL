#Xbox Controller
# This is the section from the manipulator system that handles the xbox controller:


'''
Connect to USB Xbox controller .
Left Trigger = Negative Z
Right Trigger = Positive Z
Left Joystick = 360 XY
Right joystick rolling
Back button = Disconnect Xbox Controller
'''

def Handle_Xbox():
    Text_Box.insert(tk.END, "XBOX Connected\n")
    Text_Box.see("end")
    joy = xbox.Joystick() # Instantiate the controller

    # while the back button is 0(i.e not pressed) do all of the below
    while not joy.Back():

        # Left analog stick (x axis range from -1 to 1, y axis range from -1 to 1)
        #to avoid divide by zero error in arctan

        if joy.leftX() == 0 and joy.leftY() == 0:
           pass
        else:
            Left_Joy_Direction = (180/np.pi)*np.arctan2(joy.leftY() ,joy.leftX())
            Move_Arrow(round(Left_Joy_Direction,2))
            Text_Box.insert(tk.END, str(round(joy.leftX(),2)) + str(round(joy.leftY(),2)) +"\n")
            Text_Box.see("end")
            Coil1.value = round(joy.leftY(),2)*scale  #converging
            Coil2.value = round(joy.leftX(),2)*scale  #converging
            Coil3.value = -round(joy.leftY(),2)*scale #diverging
            Coil4.value = -round(joy.leftX(),2)*scale #diverging

        #Udate Z field
        '''
        Positive Z: motor1 = +, motor2 = -
        Negative Z: motor1 = -, motor2 = +
        '''

        Pos_Z_Field = round(joy.rightTrigger(),2) #read value from right trigger and assing it to positive z field
        Neg_Z_Field = -round(joy.leftTrigger(),2) #read value from left jostick and assign it to negative z field

        if Pos_Z_Field > 0.01 and Neg_Z_Field == 0:
            #Output positive Z Field
            Coil5.value = round(Pos_Z_Field,2)*scale
            Coil6.value = -round(Pos_Z_Field,2)*scale
            #Display joystick readings in entry box
            Z_Strength_Entry.delete(0, tk.END)
            Z_Strength_Entry.insert(0, str(Pos_Z_Field)) #print 0 -> +1 as positive Z

        elif Neg_Z_Field < -0.01 and Pos_Z_Field == 0:
            #Output negative Z Field
            Coil5.value = round(Neg_Z_Field,2)*scale
            Coil6.value = -round(Neg_Z_Field,2)*scale
            #Display joystick readings in entry box
            Z_Strength_Entry.delete(0, tk.END)
            Z_Strength_Entry.insert(0, str(Neg_Z_Field)) #print 0 -> -1 as negatie Z

        else:
            #Zero everything if neither trigger is activated
            Z_Strength_Entry.delete(0, tk.END)
            Z_Strength_Entry.insert(0, str(Neg_Z_Field)) #print 0 -> -1 as negatie Z
            Coil5.value = 0*scale
            Coil6.value = 0*scale           

        '''
        #RIGHT ANALOG STICK FOR ROLLING
        if joy.rightX() == 0 and joy.rightY() == 0:
           pass
        else:
            Right_Joy_Direction = (180/np.pi)*np.arctan2(joy.rightY() ,joy.rightX())
            Move_Arrow(round(Right_Joy_Direction,2))
            Text_Box.insert(tk.END, str(round(joy.rightX(),2)) + str(round(joy.rightY(),2)) +"\n")
            Text_Box.see("end")
            A = float(Duty_Cycle) #amplitude of rotating magetnic field
            alpha = (Right_Joy_Direction-90) * (np.pi/180)  # yaw angle converted to radians
            gamma = float(Gamma_Entry.get()) * (np.pi/180)  # pitch angle converted to radians
            omega = 2*np.pi* float(Rot_Freq_Entry.get())  #angular velocity of rotating field defined from input from Rotating Frequency Entry
            period = (2*np.pi)/omega  #time it takes for one cycle
            interval = 1/50   # 1 / # discrete nodes of sine wave
            #This is off
            if joy.rightX or joy.rightY > 0.1:
                start = time.time()
                while joy.rightX or joy.rightY > 0.1:
                    tp = time.time() - start
                    Bx = A * ( (np.cos(gamma) * np.cos(alpha) * np.cos(omega*tp)) + (np.sin(alpha) * np.sin(omega*tp)))
                    By = A * ( (-np.cos(gamma) * np.sin(alpha) * np.cos(omega*tp)) + (np.cos(alpha) * np.sin(omega*tp)))
                    Bz = A * np.sin(gamma) * np.cos(omega*tp)
                    Coil1.value =   By*scale # +Y
                    Coil2.value =   Bx*scale# +X
                    Coil3.value =  -By*scale  # -Y
                    Coil4.value =  -Bx*scale  # -X
                    Coil5.value =   Bz*scale  # +Z
                    Coil6.value =  -Bz*scale  # -Z
                    Text_Box.insert(tk.END, str(round(tp,2))+"\n")
                    Text_Box.see("end")
                    window.update()
        '''
        window.update()

    #Shut everything down when the back button is pressed
    Coil1.value = 0  #converging
    Coil2.value = 0 #converging
    Coil3.value = 0 #diverging
    Coil4.value = 0 #diverging
    #Zero Z field
    Coil5.value = 0
    Coil6.value = 0

    joy.close()
    Text_Box.insert(tk.END, "\nXBOX Disconnected")
    Text_Box.see("end")




Xbox_Button = tk.Button(master = window,text = "Xbox Controller. \n Press Back Button to Exit",
                  fg = "white",bg = "red",command = Handle_Xbox)
Xbox_Button.grid(row=4,column=0, sticky = "nswe", columnspan = 2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%