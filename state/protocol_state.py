"""
Core state definitions for the protocol generation workflow.

This implements the "blackboard" architecture where agents collaboratively
read and write to shared state throughout the workflow.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ApprovalStatus(str, Enum):
    """Possible approval states for the protocol."""
    PENDING = "pending"
    PENDING_HUMAN_REVIEW = "pending_human_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    IN_PROGRESS = "in_progress"


class SafetySeverity(str, Enum):
    """Severity levels for safety flags."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentRole(str, Enum):
    """Agent roles in the system."""
    DRAFTER = "drafter"
    SAFETY = "safety"
    CRITIC = "critic"
    SUPERVISOR = "supervisor"


# Scratchpad Entry Models (for agent collaboration)

class ScratchpadEntry(BaseModel):
    """Base model for scratchpad entries."""
    timestamp: datetime = Field(default_factory=datetime.now)
    iteration: int
    agent: AgentRole
    entry_type: str


class SafetyFlag(ScratchpadEntry):
    """Safety concern flagged by Safety Guardian."""
    entry_type: str = "safety_flag"
    severity: SafetySeverity
    issue: str
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    
    model_config = {"use_enum_values": True}


class CriticFeedback(ScratchpadEntry):
    """Quality feedback from Clinical Critic."""
    entry_type: str = "critic_feedback"
    overall_score: float = Field(ge=0.0, le=10.0)
    empathy_score: float = Field(ge=0.0, le=1.0)
    individual_scores: Dict[str, float] = Field(default_factory=dict)
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    recommendation: str
    feedback: str
    confidence: float = Field(ge=0.0, le=1.0)


class SupervisorDecision(ScratchpadEntry):
    """Decision made by Supervisor."""
    entry_type: str = "supervisor_decision"
    action: str
    reason: str
    next_agent: Optional[str] = None


class DrafterNote(ScratchpadEntry):
    """Note left by Drafter agent."""
    entry_type: str = "drafter_note"
    note: str
    word_count: int
    has_structure: bool
    addressed_feedback: List[str] = Field(default_factory=list)


# Version Tracking

class DraftVersion(BaseModel):
    """A versioned draft of the protocol."""
    version_number: int
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: AgentRole
    word_count: int
    changes_summary: Optional[str] = None
    iteration: int


# Metadata and Scoring

class MetadataScores(BaseModel):
    """Aggregated scores and metrics for the protocol."""
    safety_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall safety (1.0 = perfectly safe)")
    empathy_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Empathy and tone quality")
    clinical_accuracy_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Clinical accuracy (0-10)")
    clarity_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Clarity and accessibility")
    completeness_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Completeness of content")
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Overall quality rating")
    
    def update_from_critic(self, critic_feedback: CriticFeedback):
        """Update scores based on critic feedback."""
        self.empathy_score = critic_feedback.empathy_score
        self.overall_quality_score = critic_feedback.overall_score
        
        # Update individual scores if available
        scores = critic_feedback.individual_scores
        self.clinical_accuracy_score = scores.get("clinical accuracy", scores.get("clinical_accuracy", self.clinical_accuracy_score))
        self.clarity_score = scores.get("clarity", scores.get("clarity & accessibility", self.clarity_score))
        self.completeness_score = scores.get("completeness", self.completeness_score)
    
    def update_from_safety(self, safety_flags: List[SafetyFlag]):
        """Update safety score based on safety flags."""
        if not safety_flags:
            self.safety_score = 1.0
            return
        
        # Reduce score based on severity
        penalty = 0.0
        for flag in safety_flags:
            if flag.severity == SafetySeverity.HIGH:
                penalty += 0.3
            elif flag.severity == SafetySeverity.MEDIUM:
                penalty += 0.15
            elif flag.severity == SafetySeverity.LOW:
                penalty += 0.05
        
        self.safety_score = max(0.0, 1.0 - penalty)


# Main Protocol State (The "Blackboard")

class ProtocolState(BaseModel):
    """
    The central state object that all agents read from and write to.
    """
    
    # Core Identification
    thread_id: str = Field(description="Unique identifier for this protocol generation session")
    user_intent: str = Field(description="Original user request/intent")

    source: Literal["web", "mcp"] = Field(
        default="web", 
        description="Source of the request. 'mcp' triggers auto-approval bypass."
    )
    
    # Content Management
    current_draft: str = Field(default="", description="Current version of the protocol")
    final_approved_draft: str = Field(default="", description="Human-approved final version")
    
    # Version History
    draft_versions: List[DraftVersion] = Field(default_factory=list)
    
    # Iteration Tracking
    iteration_count: int = Field(default=0)
    max_iterations: int = Field(default=5)
    
    # The Scratchpad
    scratchpad: Dict[str, List[Any]] = Field(
        default_factory=lambda: {
            "drafter_notes": [],
            "safety_flags": [],
            "safety_checks": [],
            "critic_feedback": [],
            "supervisor_decisions": [],
        }
    )
    
    # Structured Agent Outputs
    safety_flags: List[SafetyFlag] = Field(default_factory=list)
    critic_feedbacks: List[CriticFeedback] = Field(default_factory=list)
    supervisor_decisions: List[SupervisorDecision] = Field(default_factory=list)
    drafter_notes: List[DrafterNote] = Field(default_factory=list)
    
    # Metadata and Scoring
    metadata: MetadataScores = Field(default_factory=MetadataScores)
    
    # Human-in-Loop State
    halted_at_iteration: Optional[int] = Field(default=None)
    approval_status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    human_feedback: Optional[str] = Field(default=None)
    human_edited_draft: Optional[str] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    halted_at: Optional[datetime] = Field(default=None)
    approved_at: Optional[datetime] = Field(default=None)
    
    # Workflow Control Flags
    should_halt: bool = Field(default=False)
    needs_revision: bool = Field(default=False)
    is_finalized: bool = Field(default=False)
    
    # ✅ NEW: Bypass halt for MCP requests
    bypass_halt: bool = Field(
        default=False,
        description="If True, skip halt node and go directly to finalize (for MCP)"
    )
    
    # Error Tracking
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = {"use_enum_values": True}
    
    # Helper Methods
    
    def add_draft_version(self, content: str, agent: AgentRole, changes_summary: Optional[str] = None):
        """
        Add a new version to the draft history.
        
        Args:
            content: Draft content
            agent: Agent that created this version
            changes_summary: Optional summary of changes
        """
        version = DraftVersion(
            version_number=len(self.draft_versions) + 1,
            content=content,
            created_by=agent,
            word_count=len(content.split()),
            changes_summary=changes_summary,
            iteration=self.iteration_count
        )
        self.draft_versions.append(version)
        self.current_draft = content
        self.last_modified = datetime.now()
    
    def add_safety_flag(
        self,
        severity: SafetySeverity,
        issue: str,
        recommendation: str,
        confidence: float
    ):
        """
        Add a safety flag from Safety Guardian.
        
        Args:
            severity: Severity level
            issue: Description of the issue
            recommendation: Suggested fix
            confidence: Agent's confidence
        """
        flag = SafetyFlag(
            iteration=self.iteration_count,
            agent=AgentRole.SAFETY,
            severity=severity,
            issue=issue,
            recommendation=recommendation,
            confidence=confidence
        )
        self.safety_flags.append(flag)
        
        # Also add to scratchpad for backwards compatibility
        self.scratchpad["safety_flags"].append(flag.model_dump())
        
        # Update metadata
        self.metadata.update_from_safety(self.safety_flags)
        self.last_modified = datetime.now()
    
    def add_critic_feedback(
        self,
        overall_score: float,
        empathy_score: float,
        individual_scores: Dict[str, float],
        strengths: List[str],
        improvements: List[str],
        recommendation: str,
        feedback: str,
        confidence: float
    ):
        """
        Add feedback from Clinical Critic.
        
        Args:
            overall_score: Overall quality score (0-10)
            empathy_score: Empathy score (0-1)
            individual_scores: Scores for each criterion
            strengths: List of strengths
            improvements: List of improvement areas
            recommendation: Overall recommendation
            feedback: Detailed feedback text
            confidence: Agent's confidence
        """
        critic_feedback = CriticFeedback(
            iteration=self.iteration_count,
            agent=AgentRole.CRITIC,
            overall_score=overall_score,
            empathy_score=empathy_score,
            individual_scores=individual_scores,
            strengths=strengths,
            improvements=improvements,
            recommendation=recommendation,
            feedback=feedback,
            confidence=confidence
        )
        self.critic_feedbacks.append(critic_feedback)
        
        # Also add to scratchpad
        self.scratchpad["critic_feedback"].append(critic_feedback.model_dump())
        
        # Update metadata
        self.metadata.update_from_critic(critic_feedback)
        self.last_modified = datetime.now()
    
    def add_supervisor_decision(
        self,
        action: str,
        reason: str,
        next_agent: Optional[str] = None
    ):
        """
        Record a supervisor decision.
        
        Args:
            action: Decision action
            reason: Reasoning
            next_agent: Next agent to run (if applicable)
        """
        decision = SupervisorDecision(
            iteration=self.iteration_count,
            agent=AgentRole.SUPERVISOR,
            action=action,
            reason=reason,
            next_agent=next_agent
        )
        self.supervisor_decisions.append(decision)
        
        # Also add to scratchpad
        self.scratchpad["supervisor_decisions"].append(decision.model_dump())
        self.last_modified = datetime.now()
    
    def add_drafter_note(
        self,
        note: str,
        word_count: int,
        has_structure: bool,
        addressed_feedback: List[str] = None
    ):
        """
        Add a note from the Drafter.
        
        Args:
            note: Note content
            word_count: Word count of draft
            has_structure: Whether draft has proper structure
            addressed_feedback: List of feedback items addressed
        """
        drafter_note = DrafterNote(
            iteration=self.iteration_count,
            agent=AgentRole.DRAFTER,
            note=note,
            word_count=word_count,
            has_structure=has_structure,
            addressed_feedback=addressed_feedback or []
        )
        self.drafter_notes.append(drafter_note)
        
        # Also add to scratchpad
        self.scratchpad["drafter_notes"].append(drafter_note.model_dump())
        self.last_modified = datetime.now()
    
    def halt_for_human_review(self):
        """Mark the state as halted for human review."""
        self.should_halt = True
        self.halted_at_iteration = self.iteration_count
        self.halted_at = datetime.now()
        self.approval_status = ApprovalStatus.PENDING_HUMAN_REVIEW
        self.last_modified = datetime.now()
    
    def approve(self, edited_draft: Optional[str] = None):
        """
        Approve the protocol (with optional edits).
        
        Args:
            edited_draft: Human-edited version (if any)
        """
        if edited_draft:
            self.human_edited_draft = edited_draft
            self.final_approved_draft = edited_draft
            self.approval_status = ApprovalStatus.EDITED
        else:
            self.final_approved_draft = self.current_draft
            self.approval_status = ApprovalStatus.APPROVED
        
        self.is_finalized = True
        self.approved_at = datetime.now()
        self.last_modified = datetime.now()
    
    def reject(self, feedback: str):
        """
        Reject the protocol and request revision.
        
        Args:
            feedback: Human feedback for revision
        """
        self.human_feedback = feedback
        self.approval_status = ApprovalStatus.REJECTED
        self.needs_revision = True
        self.should_halt = False
        self.last_modified = datetime.now()
    
    def increment_iteration(self):
        """Increment the iteration counter."""
        self.iteration_count += 1
        self.last_modified = datetime.now()
    
    def add_error(self, error_type: str, error_message: str, agent: Optional[str] = None):
        """
        Record an error that occurred during generation.
        
        Args:
            error_type: Type of error
            error_message: Error message
            agent: Agent where error occurred
        """
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration_count,
            "error_type": error_type,
            "message": error_message,
            "agent": agent
        })
        self.last_modified = datetime.now()
    
    def get_latest_safety_assessment(self) -> Optional[SafetyFlag]:
        """Get the most recent safety flag."""
        return self.safety_flags[-1] if self.safety_flags else None
    
    def get_latest_critic_feedback(self) -> Optional[CriticFeedback]:
        """Get the most recent critic feedback."""
        return self.critic_feedbacks[-1] if self.critic_feedbacks else None
    
    def has_blocking_safety_issues(self) -> bool:
        """Check if there are blocking safety issues."""
        if not self.safety_flags:
            return False
        
        # Check latest safety flags for high severity issues
        recent_flags = [f for f in self.safety_flags if f.iteration >= self.iteration_count - 1]
        return any(f.severity == SafetySeverity.HIGH for f in recent_flags)
    
    def has_major_quality_issues(self) -> bool:
        """Check if there are major quality issues."""
        if not self.critic_feedbacks:
            return False
        
        latest_feedback = self.get_latest_critic_feedback()
        if not latest_feedback:
            return False
        
        return "MAJOR" in latest_feedback.recommendation or latest_feedback.overall_score < 6.0
    
    def get_context_for_revision(self) -> str:
        """
        Build a context string summarizing feedback for revision.
        
        Returns:
            Context string with all relevant feedback
        """
        context_parts = []
        
        # Add safety concerns
        if self.safety_flags:
            recent_safety = [f for f in self.safety_flags if f.iteration >= self.iteration_count - 1]
            if recent_safety:
                context_parts.append("**Safety Concerns:**")
                for flag in recent_safety:
                    context_parts.append(f"- [{flag.severity.upper()}] {flag.issue}")
                    context_parts.append(f"  Recommendation: {flag.recommendation}")
        
        # Add quality feedback
        if self.critic_feedbacks:
            latest_critic = self.get_latest_critic_feedback()
            if latest_critic and latest_critic.improvements:
                context_parts.append("\n**Quality Improvements Needed:**")
                for improvement in latest_critic.improvements:
                    context_parts.append(f"- {improvement}")
        
        # Add human feedback
        if self.human_feedback:
            context_parts.append(f"\n**Human Reviewer Feedback:**\n{self.human_feedback}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Create a summary dictionary for API responses.
        
        Returns:
            Summary dictionary
        """
        return {
            "thread_id": self.thread_id,
            "user_intent": self.user_intent,
            "iteration_count": self.iteration_count,
            "approval_status": self.approval_status,
            "metadata": self.metadata.model_dump(),
            "has_draft": bool(self.current_draft),
            "safety_flags_count": len(self.safety_flags),
            "is_finalized": self.is_finalized,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
        }
