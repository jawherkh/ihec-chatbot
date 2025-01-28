from flask import Flask, render_template, jsonify, send_file
import json
from io import BytesIO

app = Flask(__name__)

# Fictive data (replace with your actual data source)
fictive_data = {
    "users": [
        {
            "id": 1,
            "name": "John Doe",
            "email": "john.doe@example.com",
            "activity": [
                {"date": "2023-10-01", "action": "login"},
                {"date": "2023-10-02", "action": "logout"}
            ]
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "activity": [
                {"date": "2023-10-01", "action": "login"},
                {"date": "2023-10-03", "action": "update_profile"}
            ]
        },
        {
            "id": 3,
            "name": "Alice Johnson",
            "email": "alice.johnson@example.com",
            "activity": [
                {"date": "2023-10-02", "action": "login"},
                {"date": "2023-10-04", "action": "logout"}
            ]
        }
    ]
}

# Endpoint to serve the HTML dashboard page
@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

# Endpoint to fetch data (called by the frontend)
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(fictive_data)

# Endpoint to download data as a JSON file
@app.route('/api/download', methods=['GET'])
def download_data():
    # Convert data to JSON string
    data_str = json.dumps(fictive_data, indent=2)
    # Create a BytesIO object to simulate a file
    file_like = BytesIO(data_str.encode('utf-8'))
    # Return the file as an attachment
    return send_file(
        file_like,
        mimetype='application/json',
        as_attachment=True,
        download_name='user_data.json'
    )

if __name__ == '__main__':
    app.run(debug=True)