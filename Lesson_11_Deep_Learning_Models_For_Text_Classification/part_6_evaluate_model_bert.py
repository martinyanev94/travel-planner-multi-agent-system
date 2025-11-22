model_bert.eval()
predictions, true_labels = [], []

for batch in test_loader:
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
        outputs = model_bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    predictions.append(logits.argmax(dim=1).detach().numpy())
    true_labels.append(labels.numpy())

# Combine results
predictions = np.concatenate(predictions)
true_labels = np.concatenate(true_labels)

print(confusion_matrix(true_labels, predictions))
print(classification_report(true_labels, predictions))
