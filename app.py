from flask import Flask, request, render_template
import os
import pickle



filename='stock_senti.pkl'
clf=pickle.load(open(filename,'rb'))
cv=pickle.load(open('tfvector.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.form.get("filesubmit"):
            
            fileitem = request.form['file']
            if fileitem.filename:
    # strip the leading path from the file name
                fn = os.path.basename(fileitem.filename)
      
   # open read and write the file into the server
                open(fn, 'wb').write(fileitem.file.read())
            
            
                data=fn.read()
            
            
        elif request.form.get("submit"):
            data=request.form['headline']
        
        
        data=[data.lower()]
        vect=cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html',prediction = my_prediction)

if __name__=='__main__':
    app.run(debug=True)