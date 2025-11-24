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

#### Gender Toggle (Logged-in Users Only)
- Button group with two options:
  - "Male Coach (For Men)" - Blue
  - "Female Coach (For Women)" - Red
- Automatically loads user's current preference
- Saves preference to database on change
- Clears chat history when switching coaches

#### Guest Users
- No gender toggle shown
- Default to male coach
- Can still use the system

## How It Works

### For Users:
1. **Login** to the chat interface
2. **See gender toggle** above the input box
3. **Select** Male Coach or Female Coach
4. **Chat** - AI responds with gender-specific expertise
5. **Switch anytime** - preference is saved

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
