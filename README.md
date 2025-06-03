## Bitcoin LSTM Forecast

This minor forecast, tries to predict the prices of BTC-USD from the years 2010-2025(yesterday). A LSTM is used through pytorch. Yet, recurrence and the exploding gradient problem persists. However, the Recurrent Neural Network does help more with sequential data that does not need ATTENTION and is indeed continuous.

### Requirements
```Bash
pip install matplotlib seaborn pandas torch torchvision torchaudio numpy yfinance scikit-learn
```
### Preprocessing the data for the LSTM

```python
bitcoin = yf.download("BTC-USD",start="2010-05-18",end="2025-06-02")['Close']

bitcoin = bitcoin.reset_index()

training = bitcoin.iloc[:,1:2].values

train_size = int(len(training)* .80)

train_data = training[:train_size]
test_data = training[train_size:]
```
![btc_closing_price_data](images/closing_prices.png)

The train test split is like any train test split, but it depends on the range of years from which the historical data draws from. It differs compared to say the usual sklearn train_test split. No, it is not like that, the test data, could be more than the training data if that old “tried and true approach” is taken. You have to feel the data to determine the proper train test split. It is really a “feel thing” when it comes to picking the right train/test size.

### The Slider and the sequence length
Do not think about the sequence length too long. Do not be mathematically pedantic. You just want to find sequence that minimize the loss. It is very rarely over 10 sequences. Do not think about that much, it will waste time.

```python
def slider(df,seq_length):
    X,y = [],[]
    for i in range(len(df) - seq_length):
        X_ = df[i:(seq_length + i)]
        y_ = df[(seq_length + i)]
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



model = LSTM(input_size=1,hidden_size=228,num_layers=1,output_size=1)
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
0  2023-04-16  30315.355469     30511.199219
1  2023-04-17  29445.044922     30518.837891
2  2023-04-18  30397.552734     30381.769531
3  2023-04-19  28822.681641     30311.355469
4  2023-04-20  28245.990234     30056.074219
5  2023-04-21  27276.912109     29595.990234
6  2023-04-22  27817.500000     28965.589844
7  2023-04-23  27591.384766     28549.705078
8  2023-04-24  27525.341797     28079.708984
9  2023-04-25  28307.597656     27858.414062
10 2023-04-26  28422.701172     27876.533203
11 2023-04-27  29473.787109     28128.470703
12 2023-04-28  29340.261719     28470.035156
13 2023-04-29  29248.488281     28875.371094
14 2023-04-30  29268.806641     29201.066406
15 2023-05-01  28091.568359     29362.935547
16 2023-05-02  28680.539062     29279.101562
17 2023-05-03  29006.308594     29048.316406
18 2023-05-04  28847.710938     28989.363281
19 2023-05-05  29534.384766     28961.216797
          Date   Actual Price  Predicted Price
758 2025-05-13  104169.812500     99994.562500
759 2025-05-14  103539.406250    100040.335938
760 2025-05-15  103744.640625    100140.320312
761 2025-05-16  103489.289062    100016.687500
762 2025-05-17  103191.085938     99947.054688
763 2025-05-18  106446.007812     99928.398438
764 2025-05-19  105606.179688    100402.000000
765 2025-05-20  106791.078125    100990.984375
766 2025-05-21  109678.078125    101526.539062
767 2025-05-22  111673.273438    102662.171875
768 2025-05-23  107287.789062    104243.289062
769 2025-05-24  107791.156250    104464.500000
770 2025-05-25  109035.382812    104352.273438
771 2025-05-26  109440.359375    104545.632812
772 2025-05-27  108994.640625    104644.234375
773 2025-05-28  107802.328125    104455.234375
774 2025-05-29  105641.757812    104409.937500
775 2025-05-30  103998.562500    103804.234375
776 2025-05-31  104638.093750    102713.500000
777 2025-06-01  105652.093750    101826.828125
```


























