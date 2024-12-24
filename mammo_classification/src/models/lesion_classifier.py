class LesionClassifier:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, images):
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy