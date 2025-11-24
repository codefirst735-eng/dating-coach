# Gender-Specific Coaching System - Implementation Summary

## Overview
Implemented a dual-coach system with separate male and female relationship coaches, each with their own PDFs and specialized prompts.

## Features Implemented

### 1. Database Changes
- **UserDB Model**: Added `gender_preference` field (default: "male")
- **GeminiFileDB Model**: Added `gender_category` field to tag PDFs as "male" or "female"

### 2. Backend Changes

#### Gender-Specific System Prompts
**Male Coach Prompt:**
- Specializes in: masculine energy, frame control, attraction triggers
- Tone: Direct, confident
- Focus: Leadership, boundaries, building value

**Female Coach Prompt:**
- Specializes in: feminine energy, emotional intelligence, relationship dynamics
- Tone: Warm, insightful, empathetic
- Focus: Red flags, standards, authentic connection, self-worth

#### Updated Endpoints
1. **`/chat`**: Now filters PDFs by user's gender preference and uses appropriate prompt
2. **`/update-profile`**: Added `gender_preference` parameter
3. **`/admin/upload-pdf`**: Added `gender_category` parameter (male/female)
4. **`/admin/list-files`**: Returns `gender_category` for each file

### 3. Admin Panel Changes

#### PDF Upload Section
- Added dropdown to select coach type (Male/Female)
- PDFs are tagged with gender category on upload
- Upload status shows which coach the PDF is for

#### File List Display
- Shows color-coded badges:
  - **Blue badge**: "Male Coach" 
  - **Red badge**: "Female Coach"
- Easy visual identification of which coach uses each PDF

### 4. Chat Interface Changes

#### Gender Toggle (All Users)
- **Logged-in users**:
  - Button group with two options
  - Preference saved to database
  - Automatically loads user's current preference
  - Clears chat history when switching coaches
  
- **Guest users**:
  - Same gender toggle interface
  - Preference saved to localStorage
  - Persists across page refreshes
  - 5 query limit still applies
  - Access to same PDFs and prompts as logged-in users

#### Guest User Enhancements
- Can toggle between Male/Female coach
- Uses gender-specific PDFs and prompts
- Gender preference stored in localStorage
- Same quality responses as logged-in users
- Only difference: 5 query limit

## How It Works

### For All Users (Logged-in & Guest):
1. **Visit chat interface**
2. **See gender toggle** above the input box
3. **Select** Male Coach or Female Coach
4. **Chat** - AI responds with gender-specific expertise
5. **Switch anytime** - preference is saved
   - Logged-in: Saved to database
   - Guest: Saved to localStorage

### Guest User Specifics:
- 5 free queries (same as before)
- Full access to gender-specific coaches
- Gender preference persists in browser
- Same PDFs and prompts as logged-in users
- Can sign up anytime to remove limit

### For Admins:
1. **Go to admin panel**
2. **Select coach type** from dropdown (Male/Female)
3. **Upload PDFs** - they're tagged for that coach
4. **View files** - see which coach uses each PDF (color-coded badges)
5. **Delete** PDFs as needed

### PDF Filtering:
- When a user chats, the system:
  1. Checks user's `gender_preference`
  2. Loads only PDFs with matching `gender_category`
  3. Uses the appropriate system prompt
  4. Generates response with gender-specific context

## Testing Instructions

### Test Male Coach:
1. Login to chat
2. Select "Male Coach (For Men)"
3. Ask: "She's playing games with me, what should I do?"
4. Expect: Response about frame, boundaries, masculine energy

### Test Female Coach:
1. Select "Female Coach (For Women)"
2. Ask: "He's being distant, should I reach out?"
3. Expect: Response about standards, red flags, self-worth

### Test Admin Panel:
1. Go to admin panel
2. Select "Male Relationship Coach"
3. Upload a PDF about male dating psychology
4. Verify it shows blue "Male Coach" badge
5. Select "Female Relationship Coach"
6. Upload a PDF about female dating psychology
7. Verify it shows red "Female Coach" badge

## Database Migration Note

⚠️ **Important**: The database schema has changed. You may need to:

1. **Delete old database** (if testing locally):
   ```bash
   rm users.db
   ```

2. **Restart backend** - it will create new tables with the new fields

3. **Re-upload PDFs** with gender categories

## Files Modified

### Backend:
- `backend/main.py` - Added gender fields, prompts, filtering logic

### Frontend:
- `frontend/admin.html` - Added gender dropdown, badge display
- `frontend/chat.html` - Added gender toggle button and logic

## Next Steps (Optional Enhancements)

1. **Add gender-specific welcome messages**
2. **Track usage stats per coach type**
3. **Allow users to set default coach in profile**
4. **Add coach-specific branding/colors**
5. **Implement coach-specific fallback responses**

## Commit Hash
**Branch**: `development`
**Commit**: `51a52a4`
**Message**: "Add gender-specific coaching system with separate PDFs and prompts for male/female coaches"

---

**Status**: ✅ Fully implemented and ready for testing on development branch
