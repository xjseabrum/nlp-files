# def confmat(frame):
#     tp = frame[1][1]
#     tn = frame[0][0]
#     fp = frame[1][0]
#     fn = frame[0][1]
    
#     tpr = tp / (tp + fn)
#     tnr = tn / (tn + fp)

#     prec = tp / (tp+fp)
#     rec = tn / (tn + fn)
#     acc = 0.5 * (tpr + tnr)
#     f1 = (2 * tp) / (2*tp + fp + fn)
#     print("Precision: " + str(round(prec, 4)) + "\n" + 
#           "Recall: " + str(round(rec, 4)) + "\n" + 
#           "bAcc: " + str(round(acc, 4)) + "\n"
#           "F1: " + str(round(f1, 4)))