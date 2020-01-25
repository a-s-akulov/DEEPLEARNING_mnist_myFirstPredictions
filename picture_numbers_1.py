"""

Deep learning model, wich uses mnist dataset to do predictions.
Features: 0.98 accuracy, Well-maintained code and user interaction.

"""
try:    
    from keras.datasets import mnist
    from keras.utils import to_categorical
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Conv2D, Flatten
    
    import sys
    import matplotlib.pyplot as plt
except BaseException as error:
    input(f"Importing modules error: '{error}'\n\nPress ENTER to exit")
    sys.exit(1)


class PictureNumbers():
    """Main class."""
    def __init__(self):
        """Load mnist data."""
        print("Author: akulov.a\n")
        self.modelFileName = 'model.h5'
        
        global X_train, y_train, X_test, y_test
        
        print("Loading mnist data...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        X_train = X_train.reshape(60000, 28, 28, 1)
        X_test = X_test.reshape(10000, 28, 28, 1)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        print("Loading complete!\n")
        


    def main(self):
        """Start function.
        
        select mode: open saved model or fit it now.
        
        """
        selectedMode = Other.selectMode("Select model's load mode:", availableValues={
                '1': f"Load model from '{self.modelFileName}' file",
                '2': f"Build, fit and save to '{self.modelFileName}' new model",
                })
        if selectedMode == "1":
            self.load_model()
        else:
            self.fit_model()
    
        self.show_predictions()
        
        selectedMode = Other.selectMode("\n\nJob is finisfed. What next?", availableValues={
                '1': f"Back to selecting mode and try again",
                '2': f"Exit",
                })
        
        if selectedMode == "1":
            print("\n\n\n\n")
            return self.main()
        else:
            return 0


    def load_model(self):
        """Loading model from file."""
        print(f"\nLoading model from file '{self.modelFileName}'...")
        global model
        try:
            model = load_model(self.modelFileName)
        except OSError as error:
            input(f"Loading model from file error: '{error}'\n\nPress ENTER to exit")
            sys.exit(1)
        else:
            print("Model successfully loaded!")


    def fit_model(self):
        """Build, fit and save in model file."""
        print("\nBuilding model...")
        global model
        
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        
        print("Fitting model...")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
        
        print(f"\Saving model to file '{self.modelFileName}'...")
        try:
            model.save(self.modelFileName)
        except OSError as error:
            input(f"Saving model in file error: '{error}'\n\nPress ENTER to exit")
            sys.exit(1)
        else:
            print("Model was builded, fitted and saved successfully!")


    def show_predictions(
            self,
            predictFrom = 0,
            predictTo = 4
        ):
        """Show model predictions."""
        print(f"\nMaking model's predictions from {predictFrom} to {predictTo} element...")
        predictionsArray = model.predict(X_test[predictFrom:predictTo])
        predictions = []
        for x in predictionsArray:
            m = 0
            mi = 0
            for i in range(10):
                if x[i] > m:
                    m = x[i]
                    mi = i
            predictions.append(mi)
        print("Predictions completed. Showing results:\n")
        for idx, predictedNumber in enumerate(predictions):
            idx_ = predictFrom + idx
            image = X_test[idx_].reshape(28, 28)
            realArray = list(y_test[idx_])
            realNumber = realArray.index(max(realArray))
            print(f"Real number: {realNumber}, predicted number: {predictedNumber}. Image:")
            plt.imshow(image)
            plt.show()
            


class OtherFncs():
    """Other general functions."""
    def selectMode(self, text: str, availableValues={}):
        """Returns one of availableValues.
        
        Show for user dialog to select one of values and returns selected value
        
        text - header's text
        
        availableValues - dict:
        {
                valueKey: valueDisplayText,
        }
        
        availableValues example:
        {
                1: "Mode 1",
                2: "Mode 2",
        } -  for selecting mode using number from 1 to 2 inclusively
        
        available kay's types: str(ex: "a", "b"), int(ex: 1, 2)
        
        NOTE: always returns string (ex: "a", "1")
        
        """
        print(f"{text}\n")
        for key in availableValues:
            if type(key) == int:
                availableValues[str(key)] = availableValues[key]
                del(availableValues[key])
                key = str(key)
            elif type(key) != str:
                raise KeyError("Invalid type '{}'of key '{}' in availableValues. Key's type must be str or int".format(
                        type(key),
                        key,
                        ))
                
            print(f"{key}: {availableValues[key]}")
        
        selectedKey = input(f"\nEnter selected key ({'/'.join(availableValues.keys())}): ")
        
        if selectedKey in availableValues:
            return selectedKey
        else:
            print("Invalid key selected!\n")
            return self.selectMode(text, availableValues=availableValues)
        
        
        
        
if __name__ == "__main__":
    Main = PictureNumbers()
    Other = OtherFncs()
    sys.exit(Main.main())