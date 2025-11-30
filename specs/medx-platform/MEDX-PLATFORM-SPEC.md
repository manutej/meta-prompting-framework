# MedX Platform - Complete Technical Specification

## Executive Summary

**MedX** is a healthcare infrastructure platform consisting of three interconnected products:
1. **MedX Pro** - Voice-first clinical documentation for doctors
2. **MedX Connect** - B2B integration network for labs and pharmacies
3. **MedX Consumer** - Patient-owned Universal Medical Profile (UMP)

### Key Differentiators
| Feature | Market Status | MedX Approach |
|---------|--------------|---------------|
| Voice-first Spanish clinical docs | Does not exist | Fine-tuned Whisper + medical NLU |
| Universal patient identity | Fragmented | Decentralized ID with data sovereignty |
| Lab + Pharmacy + Doctor network | Siloed systems | Event-driven integration platform |
| Patient data ownership | Provider-controlled | Patient-controlled with granular consent |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           MedX Platform Architecture                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│   │  MedX Pro   │      │MedX Consumer│      │MedX Connect │               │
│   │  (Doctors)  │      │ (Patients)  │      │(Labs/Pharm) │               │
│   │             │      │             │      │             │               │
│   │ • Voice UI  │      │ • UMP View  │      │ • Dashboard │               │
│   │ • Notes     │      │ • Consent   │      │ • Webhooks  │               │
│   │ • Orders    │      │ • Share     │      │ • Directory │               │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘               │
│          │                    │                    │                       │
│          └────────────────────┼────────────────────┘                       │
│                               │                                            │
│                    ┌──────────▼──────────┐                                │
│                    │    API Gateway      │                                │
│                    │   (Kong + Auth0)    │                                │
│                    └──────────┬──────────┘                                │
│                               │                                            │
│   ┌───────────────────────────┼───────────────────────────┐               │
│   │                           │                           │               │
│   ▼                           ▼                           ▼               │
│ ┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐          │
│ │Voice Service│     │  FHIR Server    │     │ Connect Service │          │
│ │ (ASR + NLU) │     │ (Core Records)  │     │  (Event Bus)    │          │
│ └──────┬──────┘     └────────┬────────┘     └────────┬────────┘          │
│        │                     │                       │                    │
│        └─────────────────────┼───────────────────────┘                    │
│                              │                                            │
│                    ┌─────────▼─────────┐                                  │
│                    │   Data Platform   │                                  │
│                    │ ┌───────────────┐ │                                  │
│                    │ │  PostgreSQL   │ │                                  │
│                    │ │ (FHIR Store)  │ │                                  │
│                    │ └───────────────┘ │                                  │
│                    │ ┌───────────────┐ │                                  │
│                    │ │    Kafka      │ │                                  │
│                    │ │ (Event Bus)   │ │                                  │
│                    │ └───────────────┘ │                                  │
│                    │ ┌───────────────┐ │                                  │
│                    │ │ Consent Ledger│ │                                  │
│                    │ │ (Hyperledger) │ │                                  │
│                    │ └───────────────┘ │                                  │
│                    └───────────────────┘                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Product 1: MedX Pro (Doctors)

### Value Proposition
> "Speak. Heal. Done." - Voice-first clinical documentation that eliminates typing.

### Core Features

#### 1. Voice-First Documentation
```
┌─────────────────────────────────────────────────────────────────┐
│                    Voice Processing Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  Audio   │──►│   ASR    │──►│   NLU    │──►│   SOAP   │     │
│  │ Capture  │   │ Spanish  │   │ Medical  │   │   Gen    │     │
│  │(On-Device)│   │ Whisper  │   │ Entities │   │          │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       │              │              │              │            │
│       │         [Streaming]    [Real-time]    [Structured]     │
│       │                             │              │            │
│       │                             ▼              ▼            │
│       │                       ┌──────────────────────┐         │
│       │                       │  Clinical Alerts     │         │
│       │                       │  • Drug interactions │         │
│       │                       │  • Critical values   │         │
│       │                       │  • Missing info      │         │
│       │                       └──────────────────────┘         │
│       │                                                         │
│       ▼                                                         │
│  [HIPAA: Audio encrypted before any network transmission]      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Clinical Documentation Types
- **SOAP Notes** - Subjective, Objective, Assessment, Plan
- **H&P** - History and Physical
- **Progress Notes** - Daily/weekly updates
- **Procedure Notes** - Surgical/procedural documentation
- **Discharge Summaries** - End-of-care documentation

#### 3. Entity Extraction (Automatic)
| Entity Type | Coding System | Example |
|-------------|---------------|---------|
| Diagnosis | ICD-10-CM | "diabetes tipo 2" → E11.9 |
| Procedure | CPT | "electrocardiograma" → 93000 |
| Medication | RxNorm | "metformina 500mg" → 6809 |
| Lab Test | LOINC | "glucosa en ayunas" → 1558-6 |
| Anatomy | SNOMED-CT | "brazo izquierdo" → 368208006 |

### MedX Pro API

```yaml
# Voice Transcription Endpoint
POST /v1/voice/transcribe
Authorization: Bearer {jwt}
Content-Type: multipart/form-data

Request:
  audio: binary (opus/webm)
  patient_id: string (optional, for context)
  note_type: enum [soap, progress, hp, procedure, discharge]
  language: "es-MX" | "es-ES" | "es-AR"

Response:
  transcript: string
  confidence: float
  entities:
    - type: "diagnosis"
      text: "dolor torácico"
      code: "R07.9"
      system: "icd-10"
      span: [45, 60]
    - type: "medication"
      text: "aspirina 100mg"
      code: "1191"
      system: "rxnorm"
      span: [120, 135]
  soap_draft:
    subjective: string
    objective: string
    assessment: string
    plan: string
  alerts:
    - type: "critical"
      message: "Possible acute coronary syndrome - recommend ECG"

---

# Save Clinical Note
POST /v1/notes
Authorization: Bearer {jwt}
Content-Type: application/fhir+json

Request: FHIR DocumentReference resource

Response:
  id: string
  status: "draft" | "final"
  warnings: []

---

# Patient Summary (with consent)
GET /v1/patients/{patient_id}/summary
Authorization: Bearer {jwt}
X-Consent-Token: {consent_jwt}

Response:
  demographics: {...}
  conditions: [...] # Active diagnoses
  medications: [...] # Current meds
  allergies: [...]
  recent_labs: [...]
  recent_visits: [...]

---

# Lab Order
POST /v1/orders/lab
Authorization: Bearer {jwt}
Content-Type: application/fhir+json

Request: FHIR ServiceRequest resource

Response:
  order_id: string
  lab_name: string
  estimated_turnaround: "24h" | "48h" | "72h"
  patient_instructions: string

---

# Prescription
POST /v1/prescriptions
Authorization: Bearer {jwt}
Content-Type: application/fhir+json

Request: FHIR MedicationRequest resource

Response:
  prescription_id: string
  pharmacy_options: [...]  # Nearby pharmacies with stock
  e_prescription_token: string  # For pharmacy pickup
```

### MedX Pro Data Model

```typescript
// Core Types
interface VoiceSession {
  id: string;
  practitioner_id: string;
  patient_id?: string;
  started_at: DateTime;
  ended_at?: DateTime;
  audio_segments: AudioSegment[];
  transcript: TranscriptResult;
  status: 'recording' | 'processing' | 'complete' | 'error';
}

interface TranscriptResult {
  full_text: string;
  segments: TranscriptSegment[];
  entities: ExtractedEntity[];
  soap_draft?: SOAPNote;
  confidence: number;
  language: 'es-MX' | 'es-ES' | 'es-AR';
}

interface ExtractedEntity {
  type: 'diagnosis' | 'medication' | 'procedure' | 'lab' | 'anatomy' | 'symptom';
  text: string;
  code: string;
  system: 'icd-10' | 'rxnorm' | 'cpt' | 'loinc' | 'snomed';
  confidence: number;
  span: [number, number];
  negated: boolean;  // "no tiene diabetes" → negated: true
}

interface SOAPNote {
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  generated_codes: {
    diagnoses: string[];  // ICD-10
    procedures: string[]; // CPT
  };
}

interface ClinicalAlert {
  type: 'critical' | 'warning' | 'info';
  category: 'drug_interaction' | 'allergy' | 'critical_value' | 'missing_info' | 'clinical_decision';
  message: string;
  evidence: string[];
  suggested_action?: string;
}
```

---

## Product 2: MedX Connect (Labs & Pharmacies)

### Value Proposition
> Healthcare entity network - Connect once, integrate everywhere.

### Core Features

#### 1. Universal Adapter System
```
┌─────────────────────────────────────────────────────────────────┐
│                   MedX Connect Integration Hub                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  External Systems              MedX Connect              MedX    │
│  ┌─────────────┐              ┌───────────┐           ┌───────┐ │
│  │  Legacy Lab │──[HL7v2]────►│  Adapter  │           │       │ │
│  │   System    │◄─[HL7v2]─────│   Layer   │           │       │ │
│  └─────────────┘              │           │──[FHIR]──►│ MedX  │ │
│  ┌─────────────┐              │  ┌─────┐  │◄─[FHIR]───│  Pro  │ │
│  │   Modern    │──[FHIR]─────►│  │Trans│  │           │       │ │
│  │   Clinic    │◄─[FHIR]──────│  │form │  │           │       │ │
│  └─────────────┘              │  └─────┘  │           │       │ │
│  ┌─────────────┐              │           │──[Events]►│       │ │
│  │  Pharmacy   │──[NCPDP]────►│           │◄─[Events]─│       │ │
│  │   Chain     │◄─[Custom]────│           │           │       │ │
│  └─────────────┘              └───────────┘           └───────┘ │
│                                                                  │
│  Supported Protocols:                                           │
│  • HL7 v2.x (ADT, ORM, ORU, RDE, RDS)                          │
│  • FHIR R4 (native)                                             │
│  • NCPDP SCRIPT (pharmacy)                                      │
│  • X12 (claims/eligibility)                                     │
│  • Custom REST/SOAP (with adapter development)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Event Bus Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     Kafka Event Topics                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CLINICAL EVENTS                                                 │
│  ├── medx.lab.order.created                                     │
│  ├── medx.lab.order.accepted                                    │
│  ├── medx.lab.specimen.received                                 │
│  ├── medx.lab.result.preliminary                                │
│  ├── medx.lab.result.final                                      │
│  ├── medx.lab.result.amended                                    │
│  │                                                               │
│  ├── medx.rx.prescribed                                         │
│  ├── medx.rx.sent_to_pharmacy                                   │
│  ├── medx.rx.filled                                             │
│  ├── medx.rx.picked_up                                          │
│  ├── medx.rx.refill_due                                         │
│  │                                                               │
│  ├── medx.appointment.requested                                 │
│  ├── medx.appointment.confirmed                                 │
│  ├── medx.appointment.checked_in                                │
│  ├── medx.appointment.completed                                 │
│  ├── medx.appointment.cancelled                                 │
│  │                                                               │
│  CONSENT EVENTS                                                  │
│  ├── medx.consent.granted                                       │
│  ├── medx.consent.revoked                                       │
│  ├── medx.consent.expired                                       │
│  │                                                               │
│  AUDIT EVENTS                                                    │
│  └── medx.audit.access_log                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3. Directory Service
Searchable directory of network participants:
- Labs (by location, tests offered, turnaround time)
- Pharmacies (by location, hours, inventory)
- Providers (by specialty, accepting patients, insurance)

### MedX Connect API

```yaml
# Register as Network Participant
POST /v1/connect/register
Authorization: Bearer {api_key}

Request:
  entity_type: "lab" | "pharmacy" | "clinic" | "hospital"
  name: string
  legal_id: string  # RFC in Mexico
  locations: Location[]
  capabilities: string[]  # e.g., ["cbc", "cmp", "urinalysis"]
  protocols_supported: ["hl7v2", "fhir"]
  webhook_url: string

Response:
  entity_id: string
  api_credentials: {...}
  onboarding_checklist: [...]

---

# Subscribe to Events
POST /v1/connect/subscriptions
Authorization: Bearer {api_key}

Request:
  events: ["lab.order.created", "rx.prescribed"]
  filter:
    geographic_radius_km: 50
    location: {lat, lng}
  webhook_url: string

Response:
  subscription_id: string

---

# Publish Event
POST /v1/connect/events
Authorization: Bearer {api_key}
Content-Type: application/json

Request:
  event_type: "lab.result.final"
  payload:
    order_id: string
    result: FHIR DiagnosticReport

Response:
  event_id: string
  delivered_to: number  # subscribers notified

---

# Search Directory
GET /v1/connect/directory/labs
Authorization: Bearer {api_key}

Query Parameters:
  lat: number
  lng: number
  radius_km: number
  tests: string[]  # LOINC codes

Response:
  labs:
    - id: string
      name: string
      distance_km: number
      tests_available: string[]
      avg_turnaround_hours: number
      rating: number
      accepts_insurance: string[]

---

# Send Lab Order (Lab receives this webhook)
POST {lab_webhook_url}
Content-Type: application/json

Payload:
  event: "lab.order.created"
  order:
    id: string
    patient: {name, dob, identifier}
    ordering_provider: {name, npi}
    tests: FHIR ServiceRequest[]
    priority: "routine" | "urgent" | "stat"
    clinical_notes: string
```

### MedX Connect Data Model

```typescript
interface NetworkEntity {
  id: string;
  type: 'lab' | 'pharmacy' | 'clinic' | 'hospital' | 'imaging_center';
  name: string;
  legal_name: string;
  tax_id: string;  // RFC in Mexico
  locations: Location[];
  capabilities: Capability[];
  protocols: Protocol[];
  status: 'pending' | 'verified' | 'active' | 'suspended';
  onboarded_at: DateTime;
}

interface Location {
  id: string;
  address: Address;
  coordinates: {lat: number; lng: number};
  hours: OperatingHours;
  phone: string;
  services: string[];
}

interface EventSubscription {
  id: string;
  entity_id: string;
  event_types: string[];
  filter: EventFilter;
  webhook_url: string;
  status: 'active' | 'paused' | 'failed';
  created_at: DateTime;
}

interface LabOrder {
  id: string;
  status: 'created' | 'sent' | 'accepted' | 'specimen_received' |
          'processing' | 'preliminary' | 'final' | 'amended' | 'cancelled';
  patient_id: string;
  ordering_provider_id: string;
  lab_id: string;
  tests: FHIRServiceRequest[];
  results?: FHIRDiagnosticReport[];
  created_at: DateTime;
  updated_at: DateTime;
}

interface Prescription {
  id: string;
  status: 'prescribed' | 'sent' | 'received' | 'filled' |
          'ready' | 'picked_up' | 'cancelled';
  patient_id: string;
  prescriber_id: string;
  pharmacy_id?: string;
  medication: FHIRMedicationRequest;
  e_prescription_token: string;
  dispensing_record?: FHIRMedicationDispense;
  created_at: DateTime;
}
```

---

## Product 3: MedX Consumer (Patients)

### Value Proposition
> Own your health data. Share it on your terms.

### Core Features

#### 1. Universal Medical Profile (UMP)
```
┌─────────────────────────────────────────────────────────────────┐
│                  Universal Medical Profile (UMP)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  IDENTITY LAYER                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  │
│  │  │     DID     │  │  Biometric  │  │  Recovery   │        │  │
│  │  │ did:medx:x  │  │  (Optional) │  │  Guardians  │        │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  DATA LAYER (FHIR R4 Resources)                           │  │
│  │                                                            │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │  │
│  │  │Demographics│ │ Conditions │ │ Allergies  │            │  │
│  │  │ (Patient)  │ │ (Condition)│ │(AllergyInt)│            │  │
│  │  └────────────┘ └────────────┘ └────────────┘            │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │  │
│  │  │Medications │ │  Lab Hx    │ │ Immuniz.   │            │  │
│  │  │(MedRequest)│ │(DiagReport)│ │(Immunizatn)│            │  │
│  │  └────────────┘ └────────────┘ └────────────┘            │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │  │
│  │  │ Procedures │ │  Imaging   │ │  Visits    │            │  │
│  │  │(Procedure) │ │(ImagStudy) │ │(Encounter) │            │  │
│  │  └────────────┘ └────────────┘ └────────────┘            │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  CONSENT LAYER                                             │  │
│  │                                                            │  │
│  │  Consent Record:                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Grantee: Dr. García (Practitioner/123)             │  │  │
│  │  │  Resources: [Conditions, Medications, Labs]         │  │  │
│  │  │  Purpose: "Treatment"                               │  │  │
│  │  │  Expires: 2024-12-31                                │  │  │
│  │  │  Status: Active                                     │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  Access Log (Immutable):                                   │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  2024-01-15 10:30 - Dr. García viewed Medications   │  │  │
│  │  │  2024-01-15 10:31 - Dr. García viewed Lab Results   │  │  │
│  │  │  2024-01-14 14:00 - Lab XYZ added new result        │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Consent Management
Granular, patient-controlled access:

| Dimension | Options |
|-----------|---------|
| **Who** | Specific provider, organization, role-based |
| **What** | All data, specific resource types, specific records |
| **Why** | Treatment, payment, research, emergency |
| **When** | Time-limited, until revoked, one-time |
| **How** | View only, download, share further |

#### 3. Data Portability
- **Export**: Download complete UMP as FHIR bundle (JSON/XML)
- **Share**: Generate time-limited share links or QR codes
- **Transfer**: Move data to another provider or platform
- **Delete**: Right to deletion (with audit trail retention)

### MedX Consumer API

```yaml
# Get My Profile (Full UMP)
GET /v1/me/profile
Authorization: Bearer {patient_jwt}

Response:
  ump:
    id: string
    did: string
    demographics: FHIRPatient
    conditions: FHIRCondition[]
    medications: FHIRMedicationRequest[]
    allergies: FHIRAllergyIntolerance[]
    labs: FHIRDiagnosticReport[]
    immunizations: FHIRImmunization[]
    procedures: FHIRProcedure[]
    encounters: FHIREncounter[]
  stats:
    total_records: number
    last_updated: DateTime
    active_consents: number

---

# Grant Consent
POST /v1/me/consent
Authorization: Bearer {patient_jwt}

Request:
  grantee:
    type: "practitioner" | "organization"
    id: string
    name: string  # for display
  resources: ["Condition", "MedicationRequest", "DiagnosticReport"]
  purpose: "treatment" | "payment" | "research" | "emergency"
  expires_at: DateTime | null  # null = until revoked
  allow_sharing: boolean  # can grantee share with others?

Response:
  consent_id: string
  consent_token: string  # JWT for grantee to use
  qr_code: string  # base64 QR for in-person sharing

---

# Revoke Consent
DELETE /v1/me/consent/{consent_id}
Authorization: Bearer {patient_jwt}

Response:
  revoked: true
  effective_immediately: true

---

# View Access Log
GET /v1/me/access-log
Authorization: Bearer {patient_jwt}

Query Parameters:
  from: DateTime
  to: DateTime
  accessor_id: string (optional)

Response:
  entries:
    - timestamp: DateTime
      accessor:
        type: "practitioner"
        id: string
        name: string
      action: "view" | "download" | "share"
      resources_accessed: string[]
      consent_id: string
      ip_address: string (masked)

---

# Generate Share Link
POST /v1/me/share
Authorization: Bearer {patient_jwt}

Request:
  recipient_type: "link" | "email" | "qr"
  recipient_email: string (if email)
  resources: string[]  # resource types or specific IDs
  expires_in: "1h" | "24h" | "7d" | "30d"
  pin_protected: boolean

Response:
  share_url: string
  qr_code: string (if qr)
  expires_at: DateTime
  pin: string (if pin_protected, shown once)

---

# Export My Data
POST /v1/me/export
Authorization: Bearer {patient_jwt}

Request:
  format: "fhir-json" | "fhir-xml" | "pdf" | "c-cda"
  resources: string[] | "all"
  date_range:
    from: DateTime
    to: DateTime

Response:
  export_id: string
  status: "processing"
  download_url: string (available when ready)
  estimated_time_seconds: number

---

# Delete My Account
DELETE /v1/me
Authorization: Bearer {patient_jwt}
X-Confirm: "DELETE_MY_ACCOUNT"

Response:
  deletion_scheduled: true
  grace_period_days: 30
  audit_retained: true  # legal requirement
```

### MedX Consumer Data Model

```typescript
interface UniversalMedicalProfile {
  id: string;
  did: string;  // Decentralized Identifier
  created_at: DateTime;
  updated_at: DateTime;

  // FHIR Resources
  patient: FHIRPatient;
  conditions: FHIRCondition[];
  medications: FHIRMedicationRequest[];
  allergies: FHIRAllergyIntolerance[];
  diagnosticReports: FHIRDiagnosticReport[];
  immunizations: FHIRImmunization[];
  procedures: FHIRProcedure[];
  encounters: FHIREncounter[];
  documents: FHIRDocumentReference[];

  // MedX Extensions
  emergencyContacts: EmergencyContact[];
  advanceDirectives: AdvanceDirective[];
  insuranceInfo: InsuranceInfo[];
}

interface Consent {
  id: string;
  patient_id: string;
  grantee: {
    type: 'practitioner' | 'organization' | 'related_person';
    id: string;
    name: string;
  };
  resources: ResourceScope[];
  purpose: 'treatment' | 'payment' | 'operations' | 'research' | 'emergency';
  status: 'active' | 'revoked' | 'expired';
  created_at: DateTime;
  expires_at: DateTime | null;
  revoked_at: DateTime | null;
  allow_sharing: boolean;
}

interface ResourceScope {
  type: 'all' | 'resource_type' | 'specific';
  resource_type?: string;  // e.g., "Condition"
  resource_ids?: string[]; // specific resource IDs
  date_range?: {from: DateTime; to: DateTime};
}

interface AccessLogEntry {
  id: string;
  patient_id: string;
  timestamp: DateTime;
  accessor: {
    type: string;
    id: string;
    name: string;
    organization?: string;
  };
  action: 'view' | 'download' | 'share' | 'export';
  resources_accessed: string[];
  consent_id: string;
  ip_address_hash: string;
  user_agent_hash: string;
  // Blockchain reference for immutability
  ledger_tx_id: string;
}
```

---

## Security & Compliance Architecture

### Zero-Trust Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    Zero-Trust Security Layers                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Identity Verification                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Multi-factor authentication (TOTP, WebAuthn)         │    │
│  │  • Biometric verification (optional, mobile)            │    │
│  │  • Device trust scoring                                 │    │
│  │  • Session management with short-lived tokens           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Layer 2: Authorization (OPA)                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Role-based access control (RBAC)                     │    │
│  │  • Attribute-based access control (ABAC)                │    │
│  │  • Consent-based access control (patient grants)        │    │
│  │  • Context-aware policies (time, location, device)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Layer 3: Data Protection                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Encryption at rest (AES-256, per-tenant keys)       │    │
│  │  • Encryption in transit (TLS 1.3)                     │    │
│  │  • Field-level encryption (SSN, DOB, etc.)             │    │
│  │  • Key management (AWS KMS, customer-managed option)   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Layer 4: Audit & Monitoring                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Immutable audit logs (blockchain-backed)             │    │
│  │  • Real-time anomaly detection                          │    │
│  │  • Compliance reporting (HIPAA, GDPR)                   │    │
│  │  • Breach detection and notification                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Compliance Matrix

| Requirement | HIPAA | NOM-024 | GDPR | Implementation |
|-------------|-------|---------|------|----------------|
| Access controls | ✓ | ✓ | ✓ | RBAC + ABAC + Consent |
| Audit trails | ✓ | ✓ | ✓ | Immutable logs + blockchain |
| Encryption | ✓ | ✓ | ✓ | AES-256 + TLS 1.3 |
| Data minimization | - | - | ✓ | Granular consent scopes |
| Right to access | ✓ | ✓ | ✓ | Patient portal + export |
| Right to delete | - | - | ✓ | Soft delete + retention |
| Breach notification | ✓ | ✓ | ✓ | 72-hour automated workflow |
| BAA/DPA | ✓ | - | ✓ | Standard agreements |

---

## Technology Stack

### Infrastructure

```yaml
Cloud Infrastructure:
  Primary: AWS (us-east-1, us-west-2)
  Mexico: Azure (Mexico Central) # NOM compliance
  Kubernetes: EKS / AKS
  Service Mesh: Istio
  CDN: CloudFront + CloudFlare
  DNS: Route53 with health checks

Data Stores:
  Primary Database: PostgreSQL 15 (RDS Multi-AZ)
    - FHIR resources (JSONB)
    - Relational data
  Time-Series: TimescaleDB
    - Metrics, audit logs
  Search: Elasticsearch 8
    - Patient search, provider directory
  Cache: Redis Cluster
    - Session data, hot cache
  Object Storage: S3
    - Voice recordings (encrypted)
    - Documents, images
  Event Streaming: Apache Kafka (MSK)
    - Connect event bus
  Consent Ledger: Hyperledger Fabric
    - Immutable audit trail

Observability:
  Metrics: Prometheus + Grafana
  Logging: ELK Stack (Elasticsearch, Logstash, Kibana)
  Tracing: Jaeger
  Alerting: PagerDuty
```

### Application Stack

```yaml
Backend Services:
  API Gateway: Kong
    - Rate limiting, auth, routing
  Core API: Go 1.21
    - High-performance, type-safe
  FHIR Server: Custom Go (or HAPI FHIR)
    - FHIR R4 compliant
  ML Services: Python 3.11 + FastAPI
    - Voice processing, NLU
  Event Processor: Go + Kafka consumers

Frontend Applications:
  MedX Pro (Mobile): React Native + Expo
    - iOS and Android
    - On-device voice capture
  MedX Pro (Web): React 18 + TypeScript
    - Desktop workflow
  MedX Consumer: React Native + Expo
    - iOS and Android
  MedX Connect Portal: React 18 + TypeScript
    - B2B dashboard

Shared Libraries:
  Design System: Custom component library
  FHIR Client: TypeScript SDK
  Auth Client: Auth0 SDK wrapper
```

### AI/ML Stack

```yaml
Speech-to-Text:
  Base Model: OpenAI Whisper large-v3
  Fine-tuning:
    - Dataset: 10,000+ hours Spanish medical audio
    - Domains: Primary care, emergency, specialties
  Deployment:
    - Mobile: Whisper.cpp (on-device, quantized)
    - Cloud: vLLM on A100 GPUs (fallback)

Natural Language Understanding:
  Base Model: Llama 3.1 70B
  Fine-tuning:
    - Medical entity recognition (Spanish)
    - Clinical relationship extraction
    - SOAP note generation
  Deployment: vLLM with continuous batching

Clinical Decision Support:
  Drug Interactions: Custom model + FDA/COFEPRIS databases
  Diagnostic Suggestions: RAG over medical literature
  Critical Value Alerts: Rule engine + ML anomaly detection

Training Infrastructure:
  Framework: PyTorch 2.0
  Hardware: H100 cluster (training)
  MLOps: MLflow + Weights & Biases
```

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-4)
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Core Infrastructure + MVP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Month 1-2: Infrastructure                                       │
│  • Set up AWS/Azure multi-region                                │
│  • Deploy Kubernetes clusters                                   │
│  • Implement CI/CD pipelines                                    │
│  • Set up observability stack                                   │
│                                                                  │
│  Month 2-3: Core Services                                        │
│  • FHIR server implementation                                   │
│  • Authentication (Auth0 integration)                           │
│  • Basic API gateway                                            │
│  • Database schema + migrations                                 │
│                                                                  │
│  Month 3-4: MedX Pro MVP                                         │
│  • Voice capture (on-device)                                    │
│  • Basic transcription (Whisper API)                            │
│  • Simple note creation                                         │
│  • Provider authentication                                      │
│                                                                  │
│  Deliverable: Working voice-to-text for clinical notes          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Intelligence (Months 5-8)
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: ML Pipeline + Enhanced Features                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Month 5-6: ML Infrastructure                                    │
│  • Fine-tune Whisper on Spanish medical data                    │
│  • Train medical NLU model                                      │
│  • Deploy ML inference pipeline                                 │
│  • Implement entity extraction                                  │
│                                                                  │
│  Month 6-7: MedX Pro Enhanced                                    │
│  • SOAP note auto-generation                                    │
│  • ICD-10/CPT code suggestions                                  │
│  • Clinical alerts (basic)                                      │
│  • Patient lookup integration                                   │
│                                                                  │
│  Month 7-8: MedX Consumer MVP                                    │
│  • Patient registration + DID                                   │
│  • Basic UMP view                                               │
│  • Simple consent management                                    │
│  • Data export                                                  │
│                                                                  │
│  Deliverable: Intelligent clinical documentation +               │
│               Patient-facing app with data view                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Network (Months 9-12)
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: MedX Connect + Integration                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Month 9-10: Event Infrastructure                                │
│  • Kafka cluster deployment                                     │
│  • Event schema registry                                        │
│  • Basic adapters (FHIR, HL7v2)                                │
│  • Webhook delivery system                                      │
│                                                                  │
│  Month 10-11: Lab Integration                                    │
│  • Lab order workflow                                           │
│  • Result delivery                                              │
│  • Pilot with 2-3 labs                                          │
│                                                                  │
│  Month 11-12: Pharmacy Integration                               │
│  • E-prescription flow                                          │
│  • Pharmacy directory                                           │
│  • Pilot with 2-3 pharmacies                                    │
│                                                                  │
│  Deliverable: Working lab/pharmacy network with pilot partners  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Scale (Months 13-18)
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: Production Scale + Advanced Features                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Month 13-14: Security Hardening                                 │
│  • Penetration testing                                          │
│  • HIPAA/NOM compliance audit                                   │
│  • Consent ledger (Hyperledger)                                 │
│  • Advanced audit logging                                       │
│                                                                  │
│  Month 14-16: Advanced ML                                        │
│  • On-device Whisper deployment                                 │
│  • Specialty-specific models                                    │
│  • Clinical decision support                                    │
│  • Drug interaction checking                                    │
│                                                                  │
│  Month 16-18: Scale + Polish                                     │
│  • Performance optimization                                     │
│  • Multi-region deployment                                      │
│  • Provider onboarding tools                                    │
│  • Analytics dashboard                                          │
│                                                                  │
│  Deliverable: Production-ready platform with 100+ providers     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

### Product Metrics

| Metric | Target (Year 1) | Target (Year 2) |
|--------|-----------------|-----------------|
| **MedX Pro** | | |
| Active providers | 100 | 1,000 |
| Notes/day | 500 | 10,000 |
| Voice accuracy (Spanish medical) | 95% | 98% |
| Time saved per note | 50% | 70% |
| | | |
| **MedX Consumer** | | |
| Registered patients | 10,000 | 100,000 |
| Active UMPs | 5,000 | 50,000 |
| Consent grants/month | 1,000 | 20,000 |
| Data exports | 500 | 5,000 |
| | | |
| **MedX Connect** | | |
| Network entities | 50 | 500 |
| Events processed/day | 10,000 | 500,000 |
| Lab orders/month | 1,000 | 50,000 |
| Prescriptions/month | 2,000 | 100,000 |

### Technical Metrics

| Metric | Target |
|--------|--------|
| API latency (p99) | < 200ms |
| Voice transcription latency | < 3s |
| System uptime | 99.9% |
| Data durability | 99.999999999% |
| Security incidents | 0 critical |
| Compliance audits passed | 100% |

---

## Appendix A: FHIR Resource Extensions

### MedX Voice Extension
```json
{
  "url": "https://medx.health/fhir/StructureDefinition/voice-metadata",
  "extension": [
    {
      "url": "voice-session-id",
      "valueString": "session_abc123"
    },
    {
      "url": "language",
      "valueCode": "es-MX"
    },
    {
      "url": "confidence",
      "valueDecimal": 0.95
    },
    {
      "url": "extracted-entities",
      "extension": [
        {
          "url": "entity",
          "extension": [
            {"url": "type", "valueCode": "diagnosis"},
            {"url": "text", "valueString": "diabetes tipo 2"},
            {"url": "code", "valueCode": "E11.9"},
            {"url": "system", "valueUri": "http://hl7.org/fhir/sid/icd-10-cm"}
          ]
        }
      ]
    }
  ]
}
```

### MedX Consent Extension
```json
{
  "url": "https://medx.health/fhir/StructureDefinition/granular-consent",
  "extension": [
    {
      "url": "consent-token",
      "valueString": "jwt_token_here"
    },
    {
      "url": "resource-scope",
      "extension": [
        {"url": "resource-type", "valueCode": "Condition"},
        {"url": "date-range-start", "valueDateTime": "2023-01-01"},
        {"url": "date-range-end", "valueDateTime": "2024-12-31"}
      ]
    },
    {
      "url": "allow-sharing",
      "valueBoolean": false
    },
    {
      "url": "audit-ledger-tx",
      "valueString": "0x..."
    }
  ]
}
```

---

## Appendix B: Event Schema Examples

### Lab Result Event
```json
{
  "event_id": "evt_abc123",
  "event_type": "medx.lab.result.final",
  "timestamp": "2024-01-15T10:30:00Z",
  "source": {
    "type": "lab",
    "id": "lab_xyz",
    "name": "Laboratorios del Valle"
  },
  "payload": {
    "order_id": "order_123",
    "patient_id": "ump_456",
    "result": {
      "resourceType": "DiagnosticReport",
      "status": "final",
      "code": {
        "coding": [{
          "system": "http://loinc.org",
          "code": "2339-0",
          "display": "Glucose [Mass/volume] in Blood"
        }]
      },
      "result": [{
        "reference": "Observation/obs_789"
      }]
    }
  },
  "metadata": {
    "correlation_id": "corr_abc",
    "idempotency_key": "idem_xyz"
  }
}
```

---

*Document Version: 1.0*
*Generated via iterative meta-prompting (8 iterations)*
*Last Updated: 2024*
