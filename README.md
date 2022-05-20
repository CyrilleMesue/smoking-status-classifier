# smoking-status-classification
The data was provided by DPhi through Smoker Status Prediction Datathon. RandomForest performed best with a mean K-fold cross validation F-1 score of 84.6 %. No Feature  selection method did any better and the model ranked 4th with a score of 81.9847% on the competition's test data. Top score was 82.0804%. 

### Notes:

* Cross validation is used to observe how model generalizes
* Downsampling is used to solve class imbalance
* GridSearch is used to perform hyper parameter tunning

### Technology Stack 

1. Python  
2. Numpy
3. Pandas
4. Scikit-Learn
5. Matplotlib
6. Seaborn


### How to Run 

- Clone the repository
- Setup Virtual environment
```
$ python3 -m venv env
```
- Activate the virtual environment
```
$ source env/bin/activate
```
- Install dependencies using
```
$ pip install -r requirements.txt
```

### Contributors 

<table>
  <tr>
    <td align="center"><a href="https://github.com/CyrilleMesue"><img src="https://avatars.githubusercontent.com/CyrilleMesue" width="100px;" alt=""/><br /><sub><b>Cyrille M. NJUME</b></sub></a><br /></td>
  </tr>
</table>

### References 

[1] DPhi starter notebook: [https://dphi.tech/notebooks/3649](https://dphi.tech/notebooks/3649)


### Contact

For any feedback or queries, please reach out to [cyrillemesue@gmail.com](mailto:cyrillemesue@gmail.com).
