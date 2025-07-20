from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
import pandas as pd
import io
import re
from typing import Dict, Any
import logging
from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinSight Invoice Analyzer",
    description="Automated invoice analysis API for extracting and analyzing PDF invoice data",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_tables_from_pdf(pdf_content: bytes) -> list:
    """Extract tables from PDF using pdfplumber"""
    tables = []
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from the page
                page_tables = page.extract_tables()
                
                if page_tables:
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 0:
                            # Convert table to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append({
                                'page': page_num + 1,
                                'table': table_num + 1,
                                'data': df
                            })
                            logger.info(f"Extracted table from page {page_num + 1}, table {table_num + 1}")
                
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    
    return tables

def clean_numeric_value(value: str) -> float:
    """Clean and convert string value to float"""
    if not value or pd.isna(value):
        return 0.0
    
    # Remove common currency symbols and whitespace
    cleaned = str(value).strip()
    cleaned = re.sub(r'[$,€£¥₹\s]', '', cleaned)
    
    # Handle parentheses for negative values
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert '{value}' to float")
        return 0.0

def find_contraption_total(tables: list) -> float:
    """Find and sum the Total column values for Contraption rows"""
    total_sum = 0.0
    contraption_rows_found = 0
    
    for table_info in tables:
        df = table_info['data']
        logger.info(f"Processing table from page {table_info['page']}")
        logger.info(f"Table columns: {list(df.columns)}")
        logger.info(f"Table shape: {df.shape}")
        
        # Clean column names
        df.columns = [str(col).strip() if col else f"Col_{i}" for i, col in enumerate(df.columns)]
        
        # Find potential columns for item names and totals
        item_columns = []
        total_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['item', 'product', 'description', 'name']):
                item_columns.append(col)
            elif any(keyword in col_lower for keyword in ['total', 'amount', 'sum', 'price']):
                total_columns.append(col)
        
        # If no specific columns found, use heuristics
        if not item_columns:
            item_columns = [df.columns[0]]  # Assume first column contains items
        if not total_columns:
            # Look for the rightmost numeric-looking column
            for col in reversed(df.columns):
                sample_values = df[col].dropna().head(3)
                if len(sample_values) > 0:
                    # Check if values look numeric
                    numeric_count = 0
                    for val in sample_values:
                        try:
                            clean_numeric_value(val)
                            numeric_count += 1
                        except:
                            pass
                    if numeric_count > 0:
                        total_columns = [col]
                        break
        
        logger.info(f"Item columns: {item_columns}")
        logger.info(f"Total columns: {total_columns}")
        
        # Search for Contraption rows
        for item_col in item_columns:
            for total_col in total_columns:
                contraption_mask = df[item_col].astype(str).str.contains(
                    'contraption', case=False, na=False
                )
                contraption_rows = df[contraption_mask]
                
                if not contraption_rows.empty:
                    logger.info(f"Found {len(contraption_rows)} Contraption rows in {item_col}")
                    
                    for idx, row in contraption_rows.iterrows():
                        total_value = clean_numeric_value(row[total_col])
                        total_sum += total_value
                        contraption_rows_found += 1
                        logger.info(f"Contraption row: '{row[item_col]}' -> Total: {total_value}")
    
    logger.info(f"Total Contraption rows found: {contraption_rows_found}")
    logger.info(f"Sum of Contraption totals: {total_sum}")
    
    return total_sum

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "FinSight Invoice Analyzer API",
        "status": "active",
        "endpoints": {
            "/analyze": "POST - Upload PDF for invoice analysis"
        }
    }

@app.post("/analyze")
async def analyze_invoice(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze uploaded PDF invoice and return sum of Total column for Contraption rows
    """
    
    # Validate file type
    if not file.content_type or "pdf" not in file.content_type.lower():
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a PDF file."
        )
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        if len(pdf_content) == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        logger.info(f"Processing PDF file: {file.filename}, size: {len(pdf_content)} bytes")
        
        # Extract tables from PDF
        tables = extract_tables_from_pdf(pdf_content)
        
        if not tables:
            logger.warning("No tables found in PDF")
            return JSONResponse(
                status_code=200,
                content={
                    "sum": 0,
                    "message": "No tables found in the PDF",
                    "contraption_rows_found": 0
                }
            )
        
        # Find and sum Contraption totals
        contraption_sum = find_contraption_total(tables)
        
        return JSONResponse(
            status_code=200,
            content={
                "sum": contraption_sum,
                "tables_processed": len(tables),
                "message": "Analysis completed successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy", "service": "invoice-analyzer"}

# Mangum handler for Vercel
handler = Mangum(app)

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)