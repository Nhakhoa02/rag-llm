# CSV Indexing System

## Overview

The CSV Indexing System enhances the RAG (Retrieval-Augmented Generation) system with intelligent CSV file handling. It creates semantic indexes of CSV files by extracting the first and second rows, enabling quick relevance matching for user queries.

## Key Features

### 1. Automatic CSV Index Creation
- **First Row Extraction**: Column headers are extracted and stored as searchable content
- **Second Row Sampling**: Sample data from the second row provides context about data types and values
- **Metadata Storage**: File information, row counts, and inferred data types are preserved

### 2. Vector-Based CSV Search
- CSV indexes are stored in a dedicated `csv_indexes` collection
- Semantic search finds relevant CSV files based on user queries
- Vector embeddings enable natural language matching with column names and data

### 3. Intelligent Query Processing
- **Relevance Matching**: Finds the most relevant CSV files for a given question
- **Context-Aware Answers**: AI generates answers based on CSV structure and content
- **SQL Suggestions**: Provides appropriate SQL queries when relevant

## Architecture

### CSV Index Model
```python
class CSVIndex(BaseModel):
    csv_file_id: str
    csv_filename: str
    column_headers: List[str]      # First row
    sample_data: List[str]         # Second row
    total_rows: int
    total_columns: int
    inferred_types: Dict[str, str] # Data type inference
    index_content: str            # Formatted for vector search
```

### Storage Integration
- CSV indexes are stored in the distributed vector database
- Uses the same Qdrant-like technology across multiple nodes
- Integrates with existing collection management system

## API Endpoints

### Upload CSV Files
```http
POST /upload
Content-Type: multipart/form-data

file: CSV file
metadata: JSON metadata (optional)
```

**Response includes:**
- `csv_index_id`: ID of the created CSV index
- Standard upload response fields

### Query CSV Data
```http
POST /ask_csv
Content-Type: application/json

{
    "question": "Who are the employees in Engineering?",
    "limit": 5,
    "score_threshold": 0.5
}
```

**Response includes:**
- `answer`: AI-generated answer based on CSV structure
- `relevant_csvs`: List of relevant CSV files with scores
- `confidence`: Confidence score for the answer
- `reasoning`: Analysis details

## Usage Examples

### 1. Upload CSV Files
```python
import requests

# Upload employees.csv
with open('employees.csv', 'rb') as f:
    files = {'file': ('employees.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8000/upload', files=files)
    
# Response includes csv_index_id
result = response.json()
print(f"CSV Index ID: {result['csv_index_id']}")
```

### 2. Query CSV Data
```python
# Ask questions about CSV data
response = requests.post('http://localhost:8000/ask_csv', json={
    "question": "Which departments have the highest salaries?",
    "limit": 3
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Relevant CSVs: {len(result['relevant_csvs'])}")
```

## How It Works

### 1. CSV Processing Pipeline
```
CSV Upload → TabularProcessor → CSV Index Creation → Vector Storage
     ↓              ↓                    ↓              ↓
File Upload → Extract First/Second Row → Create Index → Store in Collection
```

### 2. Query Processing Flow
```
User Query → Search CSV Indexes → Find Relevant CSVs → Generate Answer
     ↓              ↓                    ↓              ↓
Natural Language → Vector Search → CSV Structure → AI Response
```

### 3. Index Content Generation
The system creates searchable content from CSV structure:
```
CSV File: employees.csv
Total Rows: 100
Total Columns: 6

Column Headers:
1. employee_id
2. name
3. department
4. salary
5. hire_date
6. location

Sample Data (Row 2):
1. employee_id: 1
2. name: John Doe
3. department: Engineering
4. salary: 75000
5. hire_date: 2020-01-15
6. location: New York
```

## Benefits

### 1. Semantic Understanding
- Understands natural language queries about CSV data
- Matches user intent with appropriate CSV files
- Provides context-aware responses

### 2. Scalability
- Handles multiple CSV files efficiently
- Distributed storage across multiple nodes
- Fast vector-based search

### 3. User Experience
- No need to know exact column names
- Natural language queries work
- Intelligent suggestions and explanations

## Testing

Run the test script to see the system in action:

```bash
python test_csv_indexing.py
```

This will:
1. Upload sample CSV files (employees, sales)
2. Demonstrate CSV index creation
3. Test various query types
4. Show relevance matching in action

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for AI-powered analysis
- Standard RAG system configuration applies

### Storage Configuration
- CSV indexes use the same distributed storage as other documents
- Collection name: `csv_indexes`
- Vector size: 384 (sentence-transformers)

## Future Enhancements

### Planned Features
1. **SQL Generation**: Direct SQL query generation for supported databases
2. **Data Validation**: Enhanced data type inference and validation
3. **Schema Evolution**: Handle CSV schema changes over time
4. **Advanced Analytics**: Statistical analysis and insights generation

### Integration Opportunities
1. **Database Connectors**: Direct database integration
2. **ETL Pipelines**: Automated data processing workflows
3. **Business Intelligence**: Integration with BI tools
4. **Data Governance**: Compliance and data lineage tracking 