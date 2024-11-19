# import machine
# from machine import Pin, PWM
# import time

# #pin definition
# p1 = machine.Pin(5, machine.Pin.OUT)
# p2 = machine.Pin(18, machine.Pin.OUT)
# p3 = machine.Pin(19)
# pwm = machine.PWM(p3, freq=50)

# #arduino "map" function implementation
# def convert(x, i_m, i_M, o_m, o_M):
#     return max(min(o_M, (x - i_m) * (o_M - o_m) // (i_M - i_m) + o_m), o_m)

# #easy to use function for setting motor speed and direction
# def motorSpeed(m1):
#     pwm1 = convert(abs(m1),0, 1000, 0, 1000) 
#     pwm.duty(pwm1)
#     if m1>0:
#         p1.on()
#         p2.off()
#     else:
#         p1.off()
#         p2.on()
import machine
from machine import Pin, PWM
import time

# Pin definition
p1 = machine.Pin(5, machine.Pin.OUT)  # Motor direction pin 1
p2 = machine.Pin(18, machine.Pin.OUT)  # Motor direction pin 2
p3 = machine.Pin(19)                    # PWM pin
pwm = machine.PWM(p3, freq=50)          # Set PWM frequency

# Arduino "map" function implementation
def convert(x, i_m, i_M, o_m, o_M):
    return max(min(o_M, (x - i_m) * (o_M - o_m) // (i_M - i_m) + o_m), o_m)

# Easy to use function for setting motor speed and direction
def motorSpeed(m1):
    if m1 == 0:
        p1.off()
        p2.off()
        pwm.duty(0)  # Set PWM duty to 0
        # print("Both pins OFF, motor stopped")  # Debug statement
    else:
        pwm1 = convert(abs(m1), 0, 1000, 0, 1000)
        pwm.duty(pwm1)
        # print(f"Setting PWM duty to: {pwm1}")  # Debug statement
        if m1 > 0:
            p1.on()
            p2.off()
            # print("Pin 17 (p1) ON, Pin 18 (p2) OFF")  # Debug statement
        else:
            p1.off()
            p2.on()
            # print("Pin 17 (p1) OFF, Pin 18 (p2) ON")  # Debug statement

# Example of using motorSpeed function
# This would typically be called with input from your control logic
# motorSpeed(500)  # Example to set motor speed
# motorSpeed(0)    # Stop the motor
