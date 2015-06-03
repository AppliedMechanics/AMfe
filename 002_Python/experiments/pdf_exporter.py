# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:41:08 2015

@author: johannesr
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

import time

height, width = A4
A4_landscape = (width, height)

filename_pics = '../results/stange_rigid_body_motion_20150601_110804/Bilder/Modale_ableitungen'
filename_pics = '../results/stange_20150601_154840/Bilder/mds_statisch'

filename = 'mein_pdf.pdf'
pdf = canvas.Canvas(filename, pagesize=A4_landscape)

pdf.setFont('Helvetica-Bold',20)
pdf.drawCentredString(width/2,height-(1.5*cm),'Modal derivatives')

pdf.setFont('Courier',12)
pdf.drawCentredString(width/2,height-(2.5*cm),'Filename: ' + filename_pics)



top_margin = 2*cm
bottom_margin = 1*cm
side_margin = 1*cm

no_of_mds = 7
gridwidth = width/(no_of_mds + 2)
gridheight = height/(no_of_mds + 3)
imwidth = width/(no_of_mds + 3)
imheight = height/(no_of_mds + 3)

print('Printing', no_of_mds, 'Modal derivatives')


for i in range(no_of_mds): # Zeile i
    for j in range(no_of_mds):# Spalte j
        image = filename_pics + '.'+str(i*no_of_mds + j).zfill(4)+'.png'
        x = gridwidth*(1.5+j)
        y = (no_of_mds - 0.5)*gridheight - i*gridheight
        # print(x, y)
        pdf.drawImage(image, x, y, width=imwidth, height=imheight, preserveAspectRatio=True)

for i in range(no_of_mds):
    image = filename_pics + '.'+str(no_of_mds**2 + i).zfill(4)+'.png'
    x = gridwidth*0.5
    y = (no_of_mds - 0.5)*gridheight - i*gridheight
    pdf.drawImage(image, x, y, width=imwidth, height=imheight, preserveAspectRatio=True)
    y = (no_of_mds + 0.5)*gridheight
    x = x = gridwidth*(1.5+i)
    pdf.drawImage(image, x, y, width=imwidth, height=imheight, preserveAspectRatio=True)


p = pdf.beginPath()

p.moveTo(gridwidth*1.5, gridheight*0.5)
p.lineTo(gridwidth*1.5, (no_of_mds + 1.5)*gridheight)

p.moveTo(gridwidth*0.5, (no_of_mds + 0.5)*gridheight)
p.lineTo(gridwidth*(no_of_mds + 1.5), (no_of_mds + 0.5)*gridheight)

p.moveTo(gridwidth*0.5, (no_of_mds + 1.5)*gridheight)
p.lineTo(gridwidth*1.5, (no_of_mds + 0.5)*gridheight)

p.close()

pdf.drawPath(p)
pdf.setFont('Courier',8)
pdf.drawRightString(width - 0.3*cm, 0.3*cm, 'Printed on ' + time.strftime('%d.%m.%Y %H:%M'))
print('Printing done. Saved file ', filename)
pdf.showPage()
pdf.save()


