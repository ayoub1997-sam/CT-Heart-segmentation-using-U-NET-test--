import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

smooth = 1e-15  # قيمة صغيرة لتجنب القسمة على صفر

def iou(y_true, y_pred):
    """ مقياس تقاطع الاتحاد (IoU). """
    y_true = K.flatten(y_true)  # تسطيح المصفوفة
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)  # حساب التقاطع
    union = K.sum(y_true) + K.sum(y_pred) - intersection  # حساب الاتحاد
    iou_score = (intersection + smooth) / (union + smooth)  # لتجنب القسمة على صفر
    
    return iou_score

def dice_coef(y_true, y_pred):
    """ مقياس معامل Dice. """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)
    dice_score = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    
    return dice_score

def dice_loss(y_true, y_pred):
    """ دالة خسارة Dice. """
    return 1 - dice_coef(y_true, y_pred)