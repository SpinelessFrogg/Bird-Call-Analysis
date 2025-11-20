from keras.models import load_model



def train_model(model, spec_train, spec_test, labels_train, labels_test):

    model.fit(
        spec_train, labels_train,
        validation_data=(spec_test, labels_test),
        epochs=20,
        batch_size=16,
        validation_split=0.2
    )

    model.save("birdcall_classify_model.keras")

def evaluate_model(spec_test, labels_test):
    model = load_model("birdcall_classify_model.keras")
    test_loss, test_accuracy = model.evaluate(spec_test, labels_test)
    print(f"Test accuracy: {test_accuracy:.3f}")