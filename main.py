from predict import analyze_message

print("=== AI Smart Messaging Assistant ===")

while True:

    message = input("\nEnter Message: ")

    # Exit condition
    if message.lower() == 'exit':
        print("Assistant Closed")
        break

    # Analyze message
    results = analyze_message(message)

    print("\nAnalysis Results:\n")

    for category, value in results.items():

        label, confidence = value

        print(f"{category}: {label} ({confidence}%)")