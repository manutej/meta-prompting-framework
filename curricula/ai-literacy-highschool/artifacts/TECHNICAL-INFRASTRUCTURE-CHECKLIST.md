# Technical Infrastructure Verification Checklist
## Pre-Course IT Requirements for AI Literacy Mastery

**Complete by:** 2 weeks before course start
**Owner:** IT Lead + Instructor
**Estimated time:** 2-3 hours for full verification

---

# SECTION 1: NETWORK ACCESS VERIFICATION

## Required Domains (Must Be Unblocked)

Test each domain from a student device on school network:

### Core AI Platforms

| Domain | Service | Test Method | Status |
|--------|---------|-------------|--------|
| `chat.openai.com` | ChatGPT | Load page, send test message | □ Pass □ Fail |
| `gemini.google.com` | Google Gemini | Load page, send test message | □ Pass □ Fail |
| `claude.ai` | Anthropic Claude | Load page, send test message | □ Pass □ Fail |
| `notebooklm.google.com` | NotebookLM | Upload test document | □ Pass □ Fail |

### Visual Creation

| Domain | Service | Test Method | Status |
|--------|---------|-------------|--------|
| `gamma.app` | Gamma Presentations | Create test deck | □ Pass □ Fail |
| `canva.com` | Canva | Create design, use Magic Media | □ Pass □ Fail |
| `app.leonardo.ai` | Leonardo AI | Generate test image | □ Pass □ Fail |

### Audio/Video

| Domain | Service | Test Method | Status |
|--------|---------|-------------|--------|
| `suno.com` | Suno Music | Generate test clip | □ Pass □ Fail |
| `elevenlabs.io` | ElevenLabs Voice | Generate test audio | □ Pass □ Fail |
| `capcut.com` | CapCut Editor | Load editor, import clip | □ Pass □ Fail |
| `pika.art` | Pika Video | Load page (generation optional) | □ Pass □ Fail |

### Support Services

| Domain | Service | Test Method | Status |
|--------|---------|-------------|--------|
| `podcast.adobe.com` | Adobe Podcast | Upload test audio | □ Pass □ Fail |
| `musicfx.google.com` | Google MusicFX | Generate test clip | □ Pass □ Fail |

## Firewall Exception Request Template

If any domain is blocked, submit this to IT:

```
TO: IT Security/Network Team
FROM: [Instructor Name]
RE: Firewall Exception Request for AI Literacy Course

REQUESTED DOMAINS:
[List blocked domains]

JUSTIFICATION:
These domains are required for the AI Literacy Mastery curriculum,
an approved educational program teaching students to use AI tools
ethically and effectively.

DURATION: [Course dates]

STUDENT ACCESS: [Number] students in [classroom/lab]

SUPERVISION: Instructor-supervised use during class time only

SAFETY MEASURES:
- All tools have built-in content moderation
- Students must sign acceptable use agreement
- No personal information entered into AI systems
- All AI-generated content is educational

CONTACT: [Instructor email/phone]
```

---

# SECTION 2: DEVICE REQUIREMENTS

## Minimum Specifications

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Browser** | Chrome 90+, Edge 90+, Firefox 90+ | Chrome latest |
| **RAM** | 4 GB | 8 GB |
| **Display** | 1280×720 | 1920×1080 |
| **Internet** | 5 Mbps down | 25 Mbps down |
| **Audio** | Speakers or headphones | Headphones recommended |
| **Webcam** | Not required | Optional for video |
| **Microphone** | Not required | Optional for audio |

## Device Type Compatibility

| Device | Compatibility | Notes |
|--------|---------------|-------|
| **Windows Laptop** | ✅ Full | All features work |
| **MacBook** | ✅ Full | All features work |
| **Chromebook** | ✅ Full | Use web versions only |
| **iPad** | ⚠️ Partial | Some tools limited |
| **Android Tablet** | ⚠️ Partial | Some tools limited |
| **Phone** | ❌ Not recommended | Screen too small |

## Lab/Classroom Verification

```
□ Sufficient devices for all students (1:1 ratio preferred)
□ All devices meet minimum specs
□ Power outlets available for extended sessions
□ Projector/display for instructor demos
□ Audio system for playing clips
□ Backup devices available (10% spare recommended)
```

---

# SECTION 3: ACCOUNT PROVISIONING

## Strategy Selection

Choose one approach:

### Option A: Individual Student Accounts (Recommended)

**Pros:** Student ownership, portfolios persist, more engagement
**Cons:** More setup time, age verification issues

**Pre-Course Setup (2 weeks before):**

```
□ Send account creation instructions to students
□ Specify which email to use (school vs personal)
□ Set deadline for account creation (1 week before)
□ Collect confirmation from each student
□ Verify accounts work during test session
```

**Account Creation Instructions for Students:**

```
AI LITERACY COURSE: ACCOUNT SETUP

Create accounts on these platforms BEFORE the first class:

1. ChatGPT (chat.openai.com)
   - Click "Sign up"
   - Use your [school/personal] email
   - Verify email

2. Google Gemini (gemini.google.com)
   - Sign in with your Google account
   - If using school Google, you're done

3. NotebookLM (notebooklm.google.com)
   - Sign in with same Google account

4. Canva (canva.com)
   - Sign up with school email
   - We have Education access

5. CapCut (capcut.com)
   - Sign up with email or Google

6. Suno (suno.com)
   - Sign up with email or Google

CONFIRM: Reply to this email with "ACCOUNTS READY" by [date]
```

### Option B: Shared Class Accounts

**Pros:** Simple, no student email needed
**Cons:** No individual portfolios, credential management

**Setup:**

```
□ Create one account per platform using course email
□ Set strong, shareable password
□ Document credentials securely
□ Test all accounts
□ Prepare credentials card for class distribution
```

**Shared Account Card Template:**

```
┌────────────────────────────────────┐
│    AI LITERACY CLASS ACCOUNTS      │
├────────────────────────────────────┤
│ ChatGPT: course_ai@school.edu      │
│ Password: [distribute verbally]    │
├────────────────────────────────────┤
│ Gemini: same Google account        │
│ NotebookLM: same Google account    │
├────────────────────────────────────┤
│ ⚠️ DO NOT change passwords        │
│ ⚠️ DO NOT enter personal info     │
│ ⚠️ Log out after each session     │
└────────────────────────────────────┘
```

### Option C: Instructor Demo Only

**Pros:** Minimal setup, works with restrictions
**Cons:** Least engaging, no hands-on for students

**Setup:**

```
□ Instructor creates all accounts personally
□ Large display for demos
□ Consider live-coding style (students direct, instructor types)
□ Provide handouts with example prompts/outputs
```

---

# SECTION 4: PLATFORM-SPECIFIC VERIFICATION

## ChatGPT Verification

```
□ Account created and verified
□ Can send messages (check rate limits)
□ GPT-4o access works (limited in free tier)
□ Image generation with DALL-E works
□ Web browsing feature works
□ No content blocks on educational topics
```

**Test Prompt:**
```
"Explain photosynthesis for a 10th-grade biology student.
Include the chemical equation and a kitchen analogy."
```

Expected: Educational response without blocks

## Google Gemini Verification

```
□ Signed in with appropriate account
□ Can send messages
□ Guided Learning Mode accessible (Labs > Learning Mode)
□ Image generation enabled
□ Nano Banana Pro available (may require specific access)
```

**Test Prompt:**
```
"I'm a high school student learning about World War I.
Help me understand the causes by asking me questions."
```

Expected: Socratic questioning response

## NotebookLM Verification

```
□ Can create new notebook
□ Can upload PDF (test with 10-page doc)
□ Can upload Google Doc
□ Summary generation works
□ Audio Overview generation works
□ Flashcard generation works
□ Quiz generation works
```

**Test:**
1. Upload any educational PDF
2. Click "Summarize"
3. Click "Generate Audio Overview"
4. Verify audio plays

## Canva Education Verification

```
□ Education account approved
□ Students can be added to class/team
□ Magic Media (image generation) works
□ Magic Write works
□ Export to various formats works
```

**Test:**
1. Create new design
2. Use Magic Media to generate image
3. Export as PDF

## CapCut Verification

```
□ Web version loads (capcut.com)
□ Can import video/images
□ Script-to-Video feature accessible
□ Auto-captions work
□ Export works (1080p, no watermark)
```

**Test:**
1. Open editor
2. Try Script-to-Video with: "Welcome to our class"
3. Export short clip

## Suno Verification

```
□ Account created
□ Daily credits available (50)
□ Custom mode works
□ Can download generated audio
```

**Test:**
1. Use Create with prompt: "Upbeat instrumental study music"
2. Verify generation completes
3. Verify download works

---

# SECTION 5: BACKUP SYSTEMS

## Alternative Tools Matrix

If primary tool fails, use backup:

| Primary | Backup 1 | Backup 2 |
|---------|----------|----------|
| ChatGPT | Gemini | Claude |
| Gemini | ChatGPT | Claude |
| DALL-E | Leonardo AI | Canva Magic Media |
| Nano Banana | DALL-E | Ideogram |
| Suno | Google MusicFX | - |
| ElevenLabs | Murf AI | Canva Voice |
| CapCut | Canva Video | - |
| Gamma | Canva | Beautiful.ai |

## Offline Fallback Readiness

```
□ Offline activities printed for each session
□ Example AI outputs printed for analysis
□ Discussion prompts prepared
□ Paper-based exercises ready
```

---

# SECTION 6: DATA PRIVACY VERIFICATION

## Platform Privacy Check

| Platform | Data Retention | Training Use | Student Data |
|----------|---------------|--------------|--------------|
| ChatGPT | 30 days | Opt-out available | Don't enter PII |
| Gemini | Varies | Google policies | School account safer |
| Claude | 90 days | Not used for training | Don't enter PII |
| NotebookLM | User controlled | Not shared | Documents stay private |
| Canva | User controlled | Education terms | School account |

## Student Privacy Guidelines

Distribute to students:

```
AI TOOL PRIVACY GUIDELINES

DO:
✅ Use school email when possible
✅ Use generic examples, not personal info
✅ Log out after each session
✅ Keep prompts school-appropriate

DON'T:
❌ Enter your real name in prompts
❌ Share personal addresses, phone numbers
❌ Upload personal photos
❌ Discuss other students by name
❌ Share login credentials
```

---

# SECTION 7: SIGN-OFF

## Final Verification Checklist

```
NETWORK ACCESS:
□ All required domains accessible
□ Firewall exceptions approved (if needed)

DEVICES:
□ Sufficient devices verified
□ Backup devices available
□ Audio/video equipment tested

ACCOUNTS:
□ Provisioning strategy selected
□ Accounts created/instructions sent
□ Test login successful for all platforms

BACKUPS:
□ Alternative tools identified
□ Offline activities prepared

PRIVACY:
□ Privacy guidelines distributed
□ Consent forms collected

INSTRUCTOR READINESS:
□ Instructor has all accounts
□ Instructor has tested all tools
□ Demo materials prepared
```

## Sign-Off

```
IT Lead: _________________________ Date: _________

Instructor: ______________________ Date: _________

Ready for course launch: □ YES  □ NO (see notes below)

Notes:
_________________________________________________
_________________________________________________
_________________________________________________
```

---

*Checklist Version: 1.0*
*Complete 2 weeks before course start*
