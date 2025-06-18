## Bitcoin LSTM Forecast

This minor forecast, tries to predict the prices of BTC-USD from the years 2010-2025(yesterday). A LSTM is used through pytorch. Yet, recurrence and the exploding gradient problem persists. However, the Recurrent Neural Network does help more with sequential data that does not need ATTENTION and is indeed continuous.

### Requirements
```Bash
pip install matplotlib seaborn pandas torch torchvision torchaudio numpy yfinance scikit-learn
```
### Preprocessing the data for the LSTM

```python
bitcoin = yf.download("BTC-USD",start="2010-05-17",end="2025-06-17")['Close']
bitcoin = bitcoin.shift(1)
bitcoin.dropna(inplace=True)
bitcoin = bitcoin.reset_index()

training = bitcoin.iloc[:,1:2].values

train_size = int(len(training)* 0.80)

train_data = training[:train_size]
test_data = training[train_size:]


```
![btc_closing_price_data](images/closing_prices.png)

The train test split is like any train test split, but it depends on the range of years from which the historical data draws from. It differs compared to say the usual sklearn train_test split. No, it is not like that, the test data, could be more than the training data if that old “tried and true approach” is taken. You have to feel the data to determine the proper train test split. It is really a “feel thing” when it comes to picking the right train/test size.

### The Slider and the Sequence Length
Do not think about the sequence length too long. Do not be mathematically pedantic. You just want to find sequence that minimize the loss. It is very rarely over 10 sequences. Do not think about that much, it will waste time.

```python
def slider(dataframe,seq_length):
    X,y = [],[]
    for i in range(len(dataframe) - seq_length - 1):
        X_ = dataframe[i:(seq_length + i)]
        y_ = dataframe[(seq_length + i)]
        X.append(X_)
        y.append(y_)
    return np.array(X),np.array(y)

seq_length = 1
X_train,y_train = slider(train_data,seq_length)
X_test,y_test = slider(test_data,seq_length)
```
I decided to not scale the feature range from the range(0,1). I did not think it was necceassary for the minmax scaler

### The LSTM
I used the tradional-vanilla lstm. No add on's, no GRU's, none. I feel when the target is continous, one should make it work, so I selected 500 epochs to go with the sequence length of 1.
```python
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,X):
        h0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size)
        out,_ = self.lstm(X,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out



model = LSTM(input_size=1,hidden_size=64,num_layers=1,output_size=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
loss_fn = nn.MSELoss()
epochs = 500

for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred.float(),y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        train_rsme = np.sqrt(loss_fn(y_pred,y_train))
        y_pred_test =  model(X_test)
        test_rsme = np.sqrt(loss_fn(y_pred_test,y_test))
        print(f'Epoch: {epoch}; train_RSEM: {train_rsme:.4}; Test RSME: {test_rsme:.4}')
```

### After the usual steps ...

![prediction_vs_actual](images/predicted_vs_actual_closing_prices.png)


```text
         Date  Actual Price  Predicted Price
0  2023-04-25  27525.341797     27620.935547
1  2023-04-26  28307.597656     27554.714844
2  2023-04-27  28422.701172     28339.013672
3  2023-04-28  29473.787109     28454.412109
4  2023-04-29  29340.261719     29508.042969
5  2023-04-30  29248.488281     29374.210938
6  2023-05-01  29268.806641     29282.218750
7  2023-05-02  28091.568359     29302.585938
8  2023-05-03  28680.539062     28122.433594
9  2023-05-04  29006.308594     28712.892578
10 2023-05-05  28847.710938     29039.458984
11 2023-05-06  29534.384766     28880.476562
12 2023-05-07  28904.623047     29568.783203
13 2023-05-08  28454.980469     28937.525391
14 2023-05-09  27694.275391     28486.769531
15 2023-05-10  27658.775391     27724.097656
16 2023-05-11  27621.757812     27688.505859
17 2023-05-12  27000.791016     27651.388672
18 2023-05-13  26804.992188     27028.744141
19 2023-05-14  26784.078125     26832.406250
          Date   Actual Price  Predicted Price
763 2025-05-27  109440.359375    105777.078125
764 2025-05-28  108994.640625    106135.718750
765 2025-05-29  107802.328125    105740.976562
766 2025-05-30  105641.757812    104682.742188
767 2025-05-31  103998.562500    102756.585938
768 2025-06-01  104638.093750    101284.406250
769 2025-06-02  105652.093750    101858.117188
770 2025-06-03  105881.523438    102765.835938
771 2025-06-04  105432.460938    102970.882812
772 2025-06-05  104731.976562    102569.429688
773 2025-06-06  101575.953125    101942.265625
774 2025-06-07  104390.343750     99102.609375
775 2025-06-08  105615.625000    101635.976562
776 2025-06-09  105793.640625    102733.226562
777 2025-06-10  110294.101562    102892.351562
778 2025-06-11  110257.226562    106890.468750
779 2025-06-12  108686.625000    106857.921875
780 2025-06-13  105929.054688    105467.914062
781 2025-06-14  106090.968750    103013.335938
782 2025-06-15  105472.406250    103157.960938
```




















