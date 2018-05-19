import svhn2mnist
import usps
import syn2gtrsb
import citycam
#import syndig2svhn

def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()
    elif source == 'citycam':
        return citycam.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()
    if source == 'citycam':
        return citycam.Predictor()

