
# import the necessary packages
import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask,request, render_template
app=Flask(__name__,template_folder="templates")
@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/home', methods=['GET'])
def about():
    return render_template('home.html')
@app.route('/pred',methods=['GET'])
def page():
    return render_template('upload.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("[INFO] loading model...")
    model = pickle.load(open('fdemand1.pkl', 'rb'))
    center_id=[10, 11, 13, 14, 17, 20, 23, 24, 26, 27, 29, 30, 32, 34, 36, 39, 41,
       42, 43, 50, 51, 52, 53, 55, 57, 58, 59, 61, 64, 65, 66, 67, 68, 72,
       73, 74, 75, 76, 77, 80, 81, 83, 86, 88, 89, 91, 92, 93, 94, 97, 99,
       101, 102, 104, 106, 108, 109, 110, 113, 124, 126, 129, 132, 137,
       139, 143, 145, 146, 149, 152, 153, 157, 161, 162, 174, 177, 186]
    
    
    meal_id=[1062, 1109, 1198, 1207, 1216, 1230, 1247, 1248, 1311, 1438, 1445,
       1525, 1543, 1558, 1571, 1727, 1754, 1770, 1778, 1803, 1847, 1878,
       1885, 1902, 1962, 1971, 1993, 2104, 2126, 2139, 2290, 2304, 2306,
       2322, 2444, 2490, 2492, 2494, 2539, 2569, 2577, 2581, 2631, 2640,
       2664, 2704, 2707, 2760, 2826, 2867, 2956]
    #input_features = [float(x) for x in request.form.values()]
    #features_value = [np.array(input_features)]
    #print(features_value)
    checkout_price= float(request.form["checkout_price"])
    base_price= float(request.form["base_price"])
    discount_diff= base_price-checkout_price
    print(discount_diff)
    discount_percent= (discount_diff/base_price)*100
    print(discount_percent)
    emailer_for_promotion= request.form["emailer_for_promotion"]
    print(emailer_for_promotion)
    homepage_featured= request.form["homepage_featured"]
    print(homepage_featured)
    op_area= request.form["op_area"]
    print(op_area)
    discount= [1 if discount_diff>0 else 0]
    print('this is discount')
    print(discount[0])
    center_id_inp= request.form["center_id"]
    center_id_inp= int(center_id_inp)
    center_ser=pd.Series(np.zeros(77),index=center_id)
    center_ser[center_id_inp]=1
    center_id_list=center_ser.values
    print(center_id_list)
    meal_id_inp= request.form["meal_id"]
    meal_id_inp= int(center_id_inp)
    meal_ser=pd.Series(np.zeros(51),index=meal_id)
    meal_ser[meal_id_inp]=1
    meal_id_list=meal_ser.values
    print(meal_id_list)
    region_code= int(request.form["region_code"])
    print(region_code)
    if (region_code==23):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=1,0,0,0,0,0,0,0
    if (region_code==34):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=0,1,0,0,0,0,0,0
    if (region_code==35):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=0,0,1,0,0,0,0,0
    if (region_code==56):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=0,0,0,1,0,0,0,0
    if (region_code==71):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=0,0,0,0,1,0,0,0
    if (region_code==77):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=0,0,0,0,0,1,0,0
    if (region_code==85):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=0,0,0,0,0,0,1,0
    if (region_code==93):
        rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8=0,0,0,0,0,0,0,1
    print(rc1)
    center_type= request.form["center_type"]
    if (center_type=='A'):
        ct1,ct2,ct3=1,0,0
    if (center_type=='B'):
        ct1,ct2,ct3=0,1,0
    if (center_type=='C'):
        ct1,ct2,ct3=0,0,1
    category= request.form["category"]
    if (category=='Beverages'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=1,0,0,0,0,0,0,0,0,0,0,0,0,0
    if (category=='Biryani'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,1,0,0,0,0,0,0,0,0,0,0,0,0
    if (category=='Desert'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,1,0,0,0,0,0,0,0,0,0,0,0
    if (category=='Fish'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,1,0,0,0,0,0,0,0,0,0,0
    if (category=='Extras'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,1,0,0,0,0,0,0,0,0,0
    if (category=='Other_Snacks'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,1,0,0,0,0,0,0,0,0
    if (category=='Pasta'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,1,0,0,0,0,0,0,0
    if (category=='Pizza'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,0,1,0,0,0,0,0,0
    if (category=='Rice_Bowl'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,0,0,1,0,0,0,0,0
    if (category=='Salad'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,0,0,0,1,0,0,0,0
    if (category=='Sandwich'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,0,0,0,0,1,0,0,0
    if (category=='Seafood'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,0,0,0,0,0,1,0,0
    if (category=='Soup'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,0,0,0,0,0,0,1,0
    if (category=='Starters'):
        c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14=0,0,0,0,0,0,0,0,0,0,0,0,0,1
    cuisine= int(request.form["cuisine"])
    if (cuisine==0):
        cu1,cu2,cu3,cu4=1,0,0,0
    if (cuisine==1):
        cu1,cu2,cu3,cu4=0,1,0,0
    if (cuisine==2):
        cu1,cu2,cu3,cu4=0,0,1,0
    if (cuisine==3):
        cu1,cu2,cu3,cu4=0,0,0,1
    
    city_code= request.form["city_code"]
    if (city_code=='CH1'):
        ch1,ch2,ch3,ch4=1,0,0,0
    if (city_code=='CH2'):
        ch1,ch2,ch3,ch4=0,1,0,0
    if (city_code=='CH3'):
        ch1,ch2,ch3,ch4=0,0,1,0
    if (city_code=='CH4'):
        ch1,ch2,ch3,ch4=0,0,0,1
    
    
    
    
    c=[[checkout_price,base_price,discount_percent,int(emailer_for_promotion),int(homepage_featured),
        float(op_area),discount[0],center_id_list[0],center_id_list[1],center_id_list[2],center_id_list[3],center_id_list[4],center_id_list[5],
        center_id_list[6],center_id_list[7],center_id_list[8],center_id_list[9],center_id_list[10],center_id_list[11],center_id_list[12],
        center_id_list[13],center_id_list[14],center_id_list[15],center_id_list[16],center_id_list[17],center_id_list[18],center_id_list[19],
        center_id_list[20],center_id_list[21],center_id_list[22],center_id_list[23],center_id_list[24],center_id_list[25],
        center_id_list[26],center_id_list[27],center_id_list[28],center_id_list[29],center_id_list[30],center_id_list[31],center_id_list[32],center_id_list[33],center_id_list[34],center_id_list[35],
        center_id_list[36],center_id_list[37],center_id_list[38],center_id_list[39],center_id_list[40],center_id_list[41],center_id_list[42],center_id_list[43],center_id_list[44],center_id_list[45],
        center_id_list[46],center_id_list[47],center_id_list[48],center_id_list[49],center_id_list[50],center_id_list[51],center_id_list[52],center_id_list[53],center_id_list[54],center_id_list[55],
        center_id_list[56],center_id_list[67],center_id_list[58],center_id_list[59],center_id_list[60],center_id_list[61],center_id_list[62],center_id_list[63],center_id_list[64],center_id_list[65],
        center_id_list[66],center_id_list[67],center_id_list[68],center_id_list[69],center_id_list[70],center_id_list[71],center_id_list[72],center_id_list[73],center_id_list[74],center_id_list[75],
        center_id_list[76],meal_id_list[0],meal_id_list[1],meal_id_list[2],meal_id_list[3],meal_id_list[4],meal_id_list[5],
        meal_id_list[6],meal_id_list[7],meal_id_list[8],meal_id_list[9],meal_id_list[10],meal_id_list[11],meal_id_list[12],
        meal_id_list[13],meal_id_list[14],meal_id_list[15],meal_id_list[16],meal_id_list[17],meal_id_list[18],meal_id_list[19],
        meal_id_list[20],meal_id_list[21],meal_id_list[22],meal_id_list[23],meal_id_list[24],meal_id_list[25],
        meal_id_list[26],meal_id_list[27],meal_id_list[28],meal_id_list[29],meal_id_list[30],meal_id_list[31],meal_id_list[32],meal_id_list[33],meal_id_list[34],meal_id_list[35],
        meal_id_list[36],meal_id_list[37],meal_id_list[38],meal_id_list[39],meal_id_list[40],meal_id_list[41],meal_id_list[42],meal_id_list[43],meal_id_list[44],meal_id_list[45],
        meal_id_list[46],meal_id_list[47],meal_id_list[48],meal_id_list[49],meal_id_list[50],rc1,rc2,rc3,rc4,rc5,rc6,rc7,rc8,
        ct1,ct2,ct3,c1,c2,c3,c4,c5,c,c7,c8,c9,c10,c11,c12,c13,c14,cu1,cu2,cu3,cu4,ch1,ch2,ch3,ch4]]
    c=np.array(c)
    print(c)
    print(type(c))
    
    features_name = ['homepage_featured', 'emailer_for_promotion', 'op_area', 'cuisine',
       'city_code', 'region_code', 'category']
    prediction = model.predict(c)
    output=prediction[0]   
    print("output")
    print(output)
    return render_template('upload.html', prediction_text="Number of orders:" +str(output))

    
if __name__ == '__main__':
      app.run(debug=False)
 