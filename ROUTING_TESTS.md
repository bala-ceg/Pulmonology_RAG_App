# 🧠 Intelligent Medical Router - Test Queries

## Test the intelligent routing system with these medical queries:

### 🫀 **Cardiology Queries** (Should route to Cardiology)
1. **"I'm having chest pain and shortness of breath"**
   - Keywords: chest pain, shortness of breath
   - Expected routing: Cardiology (high confidence)

2. **"What are the symptoms of heart attack?"**
   - Keywords: heart attack
   - Expected routing: Cardiology (high confidence)

3. **"My doctor said I have atrial fibrillation"**
   - Keywords: atrial fibrillation
   - Expected routing: Cardiology (high confidence)

4. **"I need information about heart failure treatment"**
   - Keywords: heart failure, treatment
   - Expected routing: Cardiology (high confidence)

### 🧠 **Neurology Queries** (Should route to Neurology)
1. **"I'm experiencing severe migraines and headaches"**
   - Keywords: migraines, headaches
   - Expected routing: Neurology (high confidence)

2. **"What are the signs of a stroke?"**
   - Keywords: stroke
   - Expected routing: Neurology (high confidence)

3. **"I have been diagnosed with epilepsy"**
   - Keywords: epilepsy
   - Expected routing: Neurology (high confidence)

4. **"My memory is getting worse, could it be Alzheimer's?"**
   - Keywords: memory, Alzheimer's
   - Expected routing: Neurology (high confidence)

### 👨‍⚕️ **Family Medicine Queries** (Should route to Family Medicine)
1. **"I need a general health checkup"**
   - Keywords: general health, checkup
   - Expected routing: Family Medicine (high confidence)

2. **"I have a cold and fever"**
   - Keywords: cold, fever
   - Expected routing: Family Medicine (high confidence)

3. **"When should I get my vaccinations?"**
   - Keywords: vaccinations
   - Expected routing: Family Medicine (high confidence)

4. **"I want to know about preventive care"**
   - Keywords: preventive care
   - Expected routing: Family Medicine (high confidence)

### 🔀 **Multi-Discipline Queries** (Should route to multiple disciplines)
1. **"I have high blood pressure and diabetes"**
   - Keywords: high blood pressure (cardiology), diabetes (family medicine)
   - Expected routing: Cardiology, Family Medicine

2. **"I had a stroke and now have heart problems"**
   - Keywords: stroke (neurology), heart problems (cardiology)
   - Expected routing: Neurology, Cardiology

3. **"I'm having headaches and chest pain"**
   - Keywords: headaches (neurology), chest pain (cardiology)
   - Expected routing: Neurology, Cardiology

### ❓ **Ambiguous/General Queries** (Should default to Family Medicine)
1. **"I don't feel well"**
   - No specific keywords
   - Expected routing: Family Medicine (default)

2. **"What's the best diet for health?"**
   - General health query
   - Expected routing: Family Medicine (default)

3. **"I need medical advice"**
   - Generic request
   - Expected routing: Family Medicine (default)

## 🎯 Testing Instructions:

1. **Start the application:**
   ```bash
   python3 main.py
   ```

2. **Open browser:** http://localhost:3000

3. **Test each query category:**
   - Type each query into the chat interface
   - Watch the "Smart Medical Router" section in the sidebar
   - Observe how queries are routed to different disciplines

4. **Expected UI Behavior:**
   - Routing status shows "Analyzing query..."
   - After response, shows which disciplines were selected
   - Displays confidence scores and routing method
   - Shows number of sources found

## 🔍 What to Look For:

### **In the Sidebar:**
- 🧠 Smart Medical Router section shows routing details
- Discipline tags appear for each query
- Confidence and source information displayed

### **In the Response:**
- **Source annotations** showing which knowledge base was used
- **Routing information** at the end of responses
- **Citations** from relevant medical disciplines
- **Confidence scores** for each source

### **Example Expected Response Format:**
```
🧠 **Intelligent Medical Assistant Response**

**Source 1: Organization KB - Cardiology** (Confidence: 95%)
[Response about chest pain from cardiology knowledge base]

**Source 2: User Uploaded Documents** (Confidence: 85%)
[Any relevant user-uploaded content]

**📚 Sources & Citations:**
📋 Cardiology: PDF: sample_knowledge.md (Page 1)

**🎯 Query Routing:** Analyzed and routed to Cardiology
```

## 🧪 Advanced Testing:

### **Test Router Logic:**
- Try medical terms vs. common language
- Test synonyms (e.g., "heart attack" vs "myocardial infarction")
- Test complex multi-symptom queries

### **Test Fallback Behavior:**
- Non-medical queries should default to Family Medicine
- Ambiguous queries should show reasoning

### **Test Performance:**
- Multiple rapid queries
- Very long/complex medical questions
- Queries with medical abbreviations

This intelligent routing system demonstrates how AI can automatically determine the most relevant medical specialties for any query, providing a much better user experience than manual discipline selection!
