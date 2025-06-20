import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from config import settings
from core.logger_utils import get_logger
from bson import ObjectId
from datetime import datetime
import mimetypes

logger = get_logger(__file__)
app = FastAPI(title="File Upload Service")

async def get_mongo_client():
    client = AsyncIOMotorClient(settings.MONGO_DATABASE_HOST)
    return client

@app.on_event("startup")
async def startup_event():
    app.state.mongo_client = await get_mongo_client()
    app.state.db = app.state.mongo_client[settings.MONGO_DATABASE_NAME]
    app.state.fs = AsyncIOMotorGridFSBucket(app.state.db)
    logger.info("Connected to MongoDB and initialized GridFS")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        # Drop the uploads collection
        await app.state.db["uploads"].drop()
        logger.info("Dropped uploads collection")

        # Delete all files in GridFS
        gridfs_collection = app.state.db["fs.files"]
        await gridfs_collection.delete_many({})
        logger.info("Deleted all GridFS files")

        # Optionally, drop GridFS chunks collection
        await app.state.db["fs.chunks"].drop()
        logger.info("Dropped GridFS chunks collection")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
    finally:
        app.state.mongo_client.close()
        logger.info("MongoDB connection closed")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        allowed_types = {
            "image/jpeg",
            "image/png",
            "application/pdf",
            "text/csv"
        }
        content_type = file.content_type
        if content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

        # Generate metadata
        file_extension = mimetypes.guess_extension(content_type) or ".bin"
        filename = file.filename or f"upload_{datetime.utcnow().isoformat()}{file_extension}"
        content = await file.read()
        metadata = {
            "original_filename": filename,
            "content_type": content_type,
            "upload_date": datetime.utcnow(),
            "size": len(content),
            "type": "file_upload"  # For CDC compatibility
        }

        # Store file in GridFS
        fs = app.state.fs
        file_id = ObjectId()
        async with await fs.open_upload_stream(
            file_id,
            metadata=metadata
        ) as gridfs_file:
            await gridfs_file.write(content)

        logger.info(f"File {filename} uploaded successfully with ID: {file_id}")

        # Store additional document in 'uploads' collection for CDC
        upload_doc = {
            "_id": file_id,
            "filename": filename,
            "content_type": content_type,
            "upload_date": datetime.utcnow(),
            "gridfs_id": file_id,
            "type": "file_upload"
        }
        await app.state.db["uploads"].insert_one(upload_doc)
        logger.info(f"Metadata for file {filename} stored in uploads collection")

        return {
            "status": "success",
            "file_id": str(file_id),
            "filename": filename,
            "content_type": content_type
        }

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)