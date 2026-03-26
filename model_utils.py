# model_utils.py
"""
Utility functions for RLHF reward model training and persistence.
Handles database operations, model saving, and training run logging.
"""

import os
from datetime import datetime
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration from environment
DB_URI = os.getenv(
    "DB_URI", 
    f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'pces_base')}"
)

engine = create_engine(DB_URI)


def save_reward_model(clf, path="reward_model.joblib"):
    """
    Save the trained reward model to disk using joblib.
    
    Args:
        clf: Trained classifier (sklearn model)
        path: File path to save the model (default: reward_model.joblib)
    
    Returns:
        str: Path where the model was saved
    """
    joblib.dump(clf, path)
    print(f"✅ Model saved to: {path}")
    return path


def log_training_run(model_path, total_samples, avg_reward, accuracy, trained_by=1001):
    """
    Log the training run details to the rlhf_reward_model_training table.
    
    Args:
        model_path: Path where the model is saved
        total_samples: Total number of training samples used
        avg_reward: Average reward score (fraction of preferred samples)
        accuracy: Model accuracy on test set
        trained_by: User ID of the person who ran training (default: 1001)
    """
    import psycopg
    from dotenv import load_dotenv
    
    load_dotenv()
    now = datetime.utcnow()
    
    # Create a descriptive model version
    model_version = f"sbert-logreg-v{now.strftime('%Y%m%d-%H%M%S')}"
    
    # Use psycopg directly to avoid SQLAlchemy sequence permission issues
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "dbname": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }
    
    # Simple INSERT without explicitly handling the sequence
    insert_sql = """
        INSERT INTO rlhf_reward_model_training
            (model_version, total_interactions, avg_reward, loss_value, accuracy, 
             trained_by, created_by, updated_by, created_dt, updated_dt)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    try:
        with psycopg.connect(**db_config, autocommit=True) as conn:
            with conn.cursor() as cur:
                # Get the maximum training_id and add 1 (workaround for sequence permission issue)
                cur.execute("SELECT COALESCE(MAX(training_id), 0) + 1 FROM rlhf_reward_model_training")
                next_id = cur.fetchone()[0]
                
                # Insert with explicit ID
                insert_with_id = """
                    INSERT INTO rlhf_reward_model_training
                        (training_id, model_version, total_interactions, avg_reward, loss_value, accuracy, 
                         trained_by, created_by, updated_by, created_dt, updated_dt)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cur.execute(insert_with_id, (
                    next_id,
                    model_version,
                    int(total_samples),
                    float(avg_reward),
                    0.0,  # loss_value
                    float(accuracy * 100.0),  # Convert to percentage
                    trained_by,
                    trained_by,  # created_by
                    trained_by,  # updated_by
                    now,
                    now
                ))
        print(f"✅ Training run logged to database with version: {model_version}")
    except Exception as e:
        print(f"⚠️ Failed to log training run to database: {e}")
        print("   (This is non-critical - model is still saved)")


def check_database_tables():
    """
    Verify that required database tables exist.
    
    Returns:
        dict: Status of each required table
    """
    status = {
        'rlhf_interactions': False,
        'rlhf_reward_model_training': False
    }
    
    try:
        with engine.connect() as conn:
            # Check rlhf_interactions
            result = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'rlhf_interactions')"
            ))
            status['rlhf_interactions'] = result.scalar()
            
            # Check rlhf_reward_model_training
            result = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'rlhf_reward_model_training')"
            ))
            status['rlhf_reward_model_training'] = result.scalar()
            
        return status
    except Exception as e:
        print(f"Error checking database tables: {e}")
        return status


if __name__ == "__main__":
    # Test database connection and table existence
    print("Testing database connection...")
    print(f"Database URI: {DB_URI.replace(os.getenv('DB_PASSWORD', 'password'), '***')}")
    
    tables = check_database_tables()
    print("\nTable Status:")
    for table, exists in tables.items():
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"  {table}: {status}")
    
    if all(tables.values()):
        print("\n✅ All required tables exist!")
    else:
        print("\n⚠️ Some required tables are missing. Please create them first.")
