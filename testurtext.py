import os
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


def read_authors_works(txt_file_path):
    authors_works = {}
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        next(file)  # 跳过标题行
        for line in file:
            if line.strip():
                parts = line.strip().split('\t')
                writer = parts[0]
                books = parts[1:4]
                # 过滤掉 "None" 值
                valid_books = [book for book in books if book != 'None']
                authors_works[writer] = valid_books
    return authors_works


def load_documents(folder):
    documents = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                words = ' '.join(jieba.cut(text))
                documents.append(words)
            labels.append(os.path.splitext(filename)[0])
    return documents, labels


def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return label_encoder, encoded_labels


def thank_you():
    messagebox.showinfo("本程序作者", "The author of the program is Ayanokoji.H.")


class TextStyleRecognizer:
    def __init__(self, data_folder, authors_works_file):
        self.scrollbar_result = None
        self.result_text = None
        self.result_frame = None
        self.scrollbar = None
        self.input_text = None
        self.input_frame = None
        self.root = None
        self.data_folder = data_folder
        self.authors_works_file = authors_works_file
        self.authors_works = read_authors_works(authors_works_file)
        self.documents, self.labels = load_documents(data_folder)
        self.label_encoder, self.labels = encode_labels(self.labels)
        self.model = self.train_model()

    def train_model(self):
        model = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        model.fit(self.documents, self.labels)
        return model

    def predict_text(self, text):
        # 使用 jieba 对输入文本进行分词
        words = ' '.join(jieba.cut(text))
        probabilities = self.model.predict_proba([words])
        adjusted_probabilities = np.maximum(probabilities - (1 / len(self.labels)), 0)
        normalized_probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)
        writer_probabilities = dict(zip(self.label_encoder.classes_, normalized_probabilities[0]))
        sorted_writers = sorted(writer_probabilities.items(), key=lambda item: item[1], reverse=True)
        top_writer = sorted_writers[0][0] if sorted_writers else "Unknown"
        top_writer_works = self.authors_works.get(top_writer, ["No works available"])
        return writer_probabilities, top_writer, top_writer_works

    def create_gui(self):
        root = tk.Tk()
        root.title("Who do you write like?")
        self.root = root  # 保存对根窗口的引用

        style = ttk.Style(root)
        style.theme_use("clam")
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.input_text = tk.Text(self.input_frame, height=10, width=50)
        self.input_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.input_frame, command=self.input_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_text.config(yscrollcommand=self.scrollbar.set)

        show_authors_button = tk.Button(root, text="显示所有作者", command=self.show_authors)
        show_authors_button.pack(side=tk.TOP)

        predict_button = tk.Button(root, text="测试文风", command=self.on_predict)
        predict_button.pack()
        clear_button = tk.Button(root, text="清除", command=self.clear_input)
        clear_button.pack(side=tk.TOP)
        show_full_similarity_button = tk.Button(root, text="与所有作家的相似度", command=self.show_full_similarity)
        show_full_similarity_button.pack(side=tk.TOP)
        info_button = tk.Button(root, text="本程序作者", command=thank_you)
        info_button.pack(side=tk.BOTTOM)
        clear2_button = tk.Button(root, text="清空结果", command=self.clear_output)
        clear2_button.pack(side=tk.BOTTOM)

        self.result_frame = ttk.Frame(self.root)
        self.result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.result_text = ScrolledText(self.result_frame, height=20, width=50)
        self.result_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.scrollbar_result = tk.Scrollbar(self.result_frame, command=self.result_text.yview)
        self.scrollbar_result.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=self.scrollbar_result.set)

        self.root.mainloop()

    def on_predict(self):
        # 从文本框中获取用户输入
        user_input = self.input_text.get("1.0", tk.END).strip()
        if user_input:
            predicted_probabilities, top_writer, top_writer_works = self.predict_text(user_input)
            self.result_text.delete("1.0", tk.END)
            sorted_probabilities = sorted(predicted_probabilities.items(), key=lambda item: item[1], reverse=True)
            messages = ["你写得最像", "也很像", "或者"]
            iteration = 1
            for writer_name, probability in sorted_probabilities:
                if iteration <= 3:
                    # 将预测结果插入到结果文本框中
                    self.result_text.insert(
                        tk.END, f"{messages[iteration - 1]} {writer_name}（{probability * 100:.2f}%）\n")
                    # 如果是第一位，显示代表作
                    if iteration == 1 and top_writer != "Unknown":
                        self.result_text.insert(tk.END, f"{top_writer}的代表作包括：{', '.join(top_writer_works)}\n")
                iteration += 1
            self.result_text.insert(tk.END, "\n")
        else:
            # 如果用户没有输入文本，则弹出警告框
            messagebox.showwarning("警告", "请输入一段文本！")

    def show_full_similarity(self):
        user_input = self.input_text.get("1.0", tk.END).strip()
        if user_input:
            # 调用 predict_text 函数获取所有作家的相似度
            words = ' '.join(jieba.cut(user_input))
            probabilities = self.model.predict_proba([words])
            adjusted_probabilities = probabilities - 1 / len(self.labels)
            adjusted_probabilities = np.maximum(adjusted_probabilities, 0)
            normalized_probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)
            writer_probabilities = dict(zip(self.label_encoder.classes_, normalized_probabilities[0]))

            # 按照降序显示所有作家的相似度
            sorted_probabilities = sorted(writer_probabilities.items(), key=lambda item: item[1], reverse=True)
            self.result_text.delete("1.0", tk.END)  # 清空输出框
            for writer_name, probability in sorted_probabilities:
                self.result_text.insert(tk.END, f"{writer_name}: {probability * 100:.2f}%\n")
        else:
            messagebox.showwarning("警告", "请输入一段文本！")

    def show_authors(self):
        authors = self.label_encoder.classes_
        authors_str = '\n'.join(authors)
        self.result_text.delete("1.0", tk.END)  # 清空输出框
        self.result_text.insert(tk.END, authors_str)

    def clear_input(self):
        self.input_text.delete("1.0", tk.END)

    def clear_output(self):
        self.result_text.delete("1.0", tk.END)


if __name__ == "__main__":
    recognizer = TextStyleRecognizer('original_data', 'information.txt')
    recognizer.create_gui()

