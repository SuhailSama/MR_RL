#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Rotating Magnetic Functionality
# outputing the rolling motion based on the incoming alpha value from the Jetson Nano
'''
add buttons and functionaity for XZ YZ and XY rotation via a rotating magnetic field
'''
def Handle_Socket():
    global Socket_Status
    Socket_Status = True
    Text_Box.insert(tk.END, "Starting Server..."+"\n")
    Text_Box.see("end")

    HOST,PORT = "169.254.132.142", 5560
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    s.bind((HOST,PORT))
    s.listen(5)
    sock_server, sock_name = s.accept()

    Text_Box.insert(tk.END, "Server Listening"+"\n")
    Text_Box.see("end")

    A = float(Duty_Cycle) #amplitude of rotating magetnic field
    gamma = float(Gamma_Entry.get()) * (np.pi/180)  # pitch angle converted to radians
    omega = 2*np.pi* float(Rot_Freq_Entry.get())  #angular velocity of rotating field defined from input from Rotating Frequency Entry

    start = time.time()
    while Socket_Status == True:
        #recieve socket message
        message = sock_server.recv(5)
        alpha = float(message.decode())

        tp = time.time() - start
        Bx =  A*( (np.cos(gamma) * np.cos(alpha) * np.cos(omega*tp)) + (np.sin(alpha) * np.sin(omega*tp)))
        By =  A*( (-np.cos(gamma) * np.sin(alpha) * np.cos(omega*tp)) + (np.cos(alpha) * np.sin(omega*tp)))
        Bz =  A*np.sin(gamma) * np.cos(omega*tp)

        Coil1.value =   By*scaley # +Y
        Coil2.value =   Bx*scalex# +X
        Coil3.value =  -By*scaley  # -Y
        Coil4.value =  -Bx*scalex  # -X
        Coil5.value =   Bz*scalez  # +Z
        Coil6.value =  -Bz*scalez  # -Z



        Move_Arrow(round(alpha*180/np.pi,2))
        Text_Box.insert(tk.END, str(round(tp,2))+"\n")
        Text_Box.see("end")
        window.update()

    sock_server.close()
    Coil1.value = 0
    Coil2.value = 0
    Coil3.value = 0
    Coil4.value = 0
    Coil5.value = 0
    Coil6.value = 0



Socket_Button = tk.Button(master = window, text = "Start Listening", width = 5, height = 1,
                             fg = "white",bg = "purple", command = Handle_Socket)
Socket_Button.grid(row=6, column=5,sticky = "nswe")
