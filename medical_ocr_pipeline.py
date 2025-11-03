import os
import json
import logging
import pytesseract
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # use open-source embeddings
import google.generativeai as genai

# ----------------------- Logging Setup -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-ocr-pipeline")

# ----------------------- Tesseract Path (Windows only) -----------------------
# Update this if Tesseract isn't in PATH
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------- OCR Engine -----------------------
class HybridOCREngine:
    def __init__(self):
        logger.info("Initializing PaddleOCR for production (no Tesseract)...")
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("✅ PaddleOCR initialized successfully")

    def extract_text_from_image(self, image_path):
        logger.info(f"Extracting text from image: {image_path}")
        try:
            paddle_results = self.paddle_ocr.ocr(image_path, cls=True)
            if not paddle_results or not paddle_results[0]:
                return "[EMPTY TEXT]"
            
            # Combine all recognized lines into one string
            text = " ".join([line[1][0] for line in paddle_results[0]])
            return text.strip()
        except Exception as e:
            logger.error(f"❌ PaddleOCR failed on {image_path}: {e}")
            return "[ERROR IN OCR]"


# ----------------------- PDF to Image -----------------------
def convert_pdf_to_images(pdf_path):
    logger.info(f"Converting PDF to images: {pdf_path}")
    try:
        images = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        base_dir = os.path.splitext(pdf_path)[0] + "_pages"
        os.makedirs(base_dir, exist_ok=True)

        for i, img in enumerate(images):
            path = os.path.join(base_dir, f"page_{i+1}.png")
            img.save(path, "PNG")
            image_paths.append(path)

        logger.info(f"✅ Converted {len(image_paths)} pages")
        return image_paths
    except Exception as e:
        logger.error(f"❌ Failed to convert PDF: {e}")
        return []

# ----------------------- Medical LLM Processor (Gemini) -----------------------
class MedicalLLMProcessor:
    def __init__(self, api_key: str):
        logger.info("Initializing Gemini LLM client...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("✅ Gemini initialized successfully")

    def extract_structured_info(self, ocr_text: str):
        try:
            prompt = (
                "You are a medical document interpreter. Extract structured information from OCR text "
                "and produce JSON in this format:\n"
                "{\n"
                "  'PatientName': str,\n"
                "  'Age': str,\n"
                "  'Gender': str,\n"
                "  'Diagnosis': str,\n"
                "  'Tests': [str],\n"
                "  'DoctorName': str,\n"
                "  'Hospital': str,\n"
                "  'Date': str\n"
                "}\n\n"
                "OCR Text:\n\n" + ocr_text
            )

            response = self.model.generate_content(prompt)
            result = response.text.strip()

            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"raw_output": result}

        except Exception as e:
            logger.error(f"❌ Gemini extraction failed: {e}")
            return {"error": str(e)}

# ----------------------- Vector DB -----------------------
class VectorDatabase:
    def __init__(self, db_path: str):
        logger.info("Initializing Vector Database...")
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        # Using lightweight open-source embeddings for local use
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        logger.info("✅ Vector DB ready")

    def store_texts(self, texts, metadatas):
        if not texts:
            logger.warning("⚠️ No texts to store in vector DB")
            return
        self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        self.vector_store.save_local(self.db_path)
        logger.info("✅ Stored extracted texts in vector DB")

# ----------------------- Pipeline -----------------------
class MedicalDocumentPipeline:
    def __init__(self, gemini_api_key: str, vector_db_path: str = "./medical_vector_db"):
        self.ocr_engine = HybridOCREngine()
        self.llm_processor = MedicalLLMProcessor(gemini_api_key)
        self.vector_db = VectorDatabase(vector_db_path)

    def process_document(self, file_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        if file_path.lower().endswith(".pdf"):
            image_paths = convert_pdf_to_images(file_path)
        else:
            image_paths = [file_path]

        all_texts, structured_outputs = [], []

        for image_path in image_paths:
            text = self.ocr_engine.extract_text_from_image(image_path)
            structured = self.llm_processor.extract_structured_info(text)

            all_texts.append(text)
            structured_outputs.append({
                "image_path": image_path,
                "structured_data": structured
            })

        self.vector_db.store_texts(all_texts, metadatas=structured_outputs)

        # Save Prover JSON
        output_json = {
            "document": os.path.basename(file_path),
            "total_pages": len(image_paths),
            "extracted_data": structured_outputs
        }

        json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_prover.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved Prover JSON: {json_path}")
        return output_json

# ----------------------- Run Locally -----------------------
if __name__ == "__main__":
    # ✅ Add your local test file path here
    file_path = r"C:\Users\mraja\Downloads\test_doc.pdf"

    # ✅ Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")

    pipeline = MedicalDocumentPipeline(
        gemini_api_key=api_key,
        vector_db_path="./medical_vector_db"
    )

    result = pipeline.process_document(file_path, "./output")

    print("\n=== Extracted Summary ===")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])
