from flask import Flask, request, jsonify, render_template, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId
import random
import string

app = Flask(__name__)

# MongoDB bağlantısı
client = MongoClient("mongodb://localhost:27017/")
db = client["flaskdb"]
collection = db["users"]

# Ana sayfa (UI)
@app.route("/")
def home():
    return render_template("index.html")

# Kullanıcı ekleme formu (UI)
@app.route("/add-user", methods=["GET"])
def add_user_form():
    return render_template("add_user.html")

# Tüm kullanıcıları listele, filtrele, sırala (UI)
@app.route("/users", methods=["GET"])
def users_page():
    query = {}
    sort = request.args.get("sort")
    search = request.args.get("search")
    if search:
        query = {"$or": [
            {"name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}}
        ]}
    users_cursor = collection.find(query)
    if sort:
        users_cursor = users_cursor.sort(sort, 1)
    users = []
    for user in users_cursor:
        user["_id"] = str(user["_id"])
        users.append(user)
    return render_template("users.html", users=users, search=search or "", sort=sort or "")

# Belirli kullanıcıyı göster (UI)
@app.route("/user/<id>", methods=["GET"])
def user_page(id):
    user = collection.find_one({"_id": ObjectId(id)})
    if user:
        user["_id"] = str(user["_id"])
        return render_template("user.html", user=user)
    return render_template("user.html", user=None, error="Kullanıcı bulunamadı!")

# Kullanıcı güncelleme formu (UI)
@app.route("/edit-user/<id>", methods=["GET"])
def edit_user_form(id):
    user = collection.find_one({"_id": ObjectId(id)})
    if user:
        user["_id"] = str(user["_id"])
        return render_template("edit_user.html", user=user)
    return redirect(url_for("users_page"))

# Kullanıcı ekleme (API)
@app.route("/api/add", methods=["POST"])
def add_user():
    data = request.json
    result = collection.insert_one(data)
    return jsonify({"inserted_id": str(result.inserted_id)})

# Kullanıcı güncelleme (API)
@app.route("/api/update/<id>", methods=["POST"])
def update_user(id):
    data = request.json
    result = collection.update_one({"_id": ObjectId(id)}, {"$set": data})
    return jsonify({"matched_count": result.matched_count, "modified_count": result.modified_count})

# Kullanıcı silme (API)
@app.route("/api/delete/<id>", methods=["POST"])
def delete_user(id):
    result = collection.delete_one({"_id": ObjectId(id)})
    return jsonify({"deleted_count": result.deleted_count})

# Tüm kullanıcıları listele (API)
@app.route("/api/users", methods=["GET"])
def get_users():
    query = {}
    sort = request.args.get("sort")
    search = request.args.get("search")
    if search:
        query = {"$or": [
            {"name": {"$regex": search, "$options": "i"}},
            {"email": {"$regex": search, "$options": "i"}}
        ]}
    users_cursor = collection.find(query)
    if sort:
        users_cursor = users_cursor.sort(sort, 1)
    users = []
    for user in users_cursor:
        user["_id"] = str(user["_id"])
        users.append(user)
    return jsonify(users)

# Belirli kullanıcıyı bul (API)
@app.route("/api/user/<id>", methods=["GET"])
def get_user(id):
    user = collection.find_one({"_id": ObjectId(id)}, {"_id": 0})
    if user:
        return jsonify(user)
    return jsonify({"error": "Kullanıcı bulunamadı!"}), 404

# Rastgele kullanıcı verisi ekleme (API)
@app.route("/generate-sample")
def generate_sample():
    names = ["Ali", "Veli", "Ayşe", "Fatma", "Mehmet", "Zeynep", "Emre", "Elif", "Ahmet", "Deniz"]
    domains = ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com"]
    users = []
    for i in range(100):
        name = random.choice(names) + ''.join(random.choices(string.ascii_lowercase, k=3))
        email = f"{name.lower()}{random.randint(1,999)}@{random.choice(domains)}"
        users.append({"name": name, "email": email})
    collection.insert_many(users)
    return f"{len(users)} adet örnek kullanıcı eklendi!"

# Dokümantasyon sayfası (UI)
@app.route("/docs")
def docs():
    return render_template("docs.html")

if __name__ == "__main__":
    app.run(debug=True)