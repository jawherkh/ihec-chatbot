# IHEC Chatbot

## Overview
The IHEC Chatbot is a web-based application designed to provide information about IHEC Carthage. It utilizes advanced natural language processing techniques to answer user queries related to the university's programs, services, events, and history.

## Features
- **Interactive Chat Interface**: Users can interact with the chatbot through a user-friendly interface.
- **Natural Language Processing**: The chatbot uses machine learning models to understand and respond to user queries.
- **Data Encryption**: User messages are encrypted for security during transmission.
- **Feedback Mechanism**: Users can provide feedback on the chatbot's responses.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: LlamaIndex, HuggingFace
- **Database**: SQLite (for session management)
- **Encryption**: AES encryption for secure data transmission

## Installation

### Prerequisites
- Python 3.x
- Node.js (for frontend dependencies, if applicable)
- Flask
- Required Python packages (listed in `requirements.txt`)

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ihec-chatbot.git
   cd ihec-chatbot
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   - Create a `.env` file in the root directory and add your Flask secret key:
     ```
     FLASK_SECRET_KEY=your_secret_key
     ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000` to access the chatbot.

## Usage
- Click on the chatbot icon to open the chat window.
- Type your question in the input field and press "Send" to receive a response.
- You can provide feedback on the chatbot's responses using the feedback buttons.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [LlamaIndex](https://github.com/jerryjliu/llama_index) for the natural language processing capabilities.
- [CryptoJS](https://cryptojs.gitbook.io/docs/) for encryption utilities.

## Contact
For any inquiries, please contact [your-email@example.com](mailto:your-email@example.com).
