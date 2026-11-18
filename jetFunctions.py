import spidev
import RPi.GPIO as GPIO
import time
import numpy as np
###Самодеятельность
def save(samples, start, finish, filename):
    with open(filename, "w") as outfile:
        outfile.write("- Blood Lab\n")
        outfile.write("- Date: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))))
        outfile.write("- Duration: {:.2f} s\n\n".format(finish - start))
        np.savetxt(outfile, np.array(samples), fmt="%d")

def read(filename):
    with open(filename) as f:
        lines = f.readlines()
    duration = float(lines[2].split()[2])
    samples = np.asarray(lines[4:], dtype=int)
    return samples, duration, len(samples)
########
########################################
#   Open, use and close SPI ADC
########################################

########################################
# Do not forget to setup GPIO pins to SPI functions!
#
# Enter the followig commands into RPi terminal:
#
# raspi-gpio get
# raspi-gpio set 9 a0
# raspi-gpio set 10 a0
# raspi-gpio set 11 a0
# raspi-gpio get
########################################

spi = spidev.SpiDev()


def initSpiAdc():
    spi.open(0, 0)
    spi.max_speed_hz = 1600000
    print("SPI for ADC has been initialized")


def deinitSpiAdc():
    spi.close()
    print("SPI cleanup finished")


def getAdc():
    adcResponse = spi.xfer2([0, 0, 0, 0])
    # print(adcResponse)
    return int(adcResponse[0] << 16 | adcResponse[1] << 8 | adcResponse[2])


########################################
#   Setup and use GPIO for step motor
########################################

directionPin = 24
enablePin = 25
stepPin = 26


def initStepMotorGpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([directionPin, enablePin, stepPin], GPIO.OUT)
    print("GPIO for step motor have been initialized")


def deinitStepMotorGpio():
    GPIO.output([directionPin, enablePin, stepPin], 0)
    GPIO.cleanup()
    print("GPIO cleanup finished")


def step():
    GPIO.output(stepPin, 0)
    time.sleep(0.005)
    GPIO.output(stepPin, 1)
    time.sleep(0.005)


def stepForward(n):
    GPIO.output(directionPin, 1)
    GPIO.output(enablePin, 1)

    for i in range(n):
        step()

    GPIO.output(enablePin, 0)


def stepBackward(n):
    GPIO.output(directionPin, 0)
    GPIO.output(enablePin, 1)

    for i in range(n):
        step()

    GPIO.output(enablePin, 0)