"""
MCP Server for Cerina Protocol Foundry using FastMCP.

Exposes the LangGraph CBT protocol workflow as:
- 1 PRIMARY TOOL: generate_cbt_protocol (the entire workflow)
- RESOURCES: For viewing generated protocols (read-only data)
"""

from mcp.server.fastmcp import FastMCP
import httpx
from typing import Optional
import asyncio
import time

mcp = FastMCP("Cerina Protocol Foundry")

API_BASE_URL = "http://localhost:8000/api"

# Store active thread IDs for resource discovery
_active_threads = []


# ============================================================================
# PRIMARY TOOL: The Single Workflow Entry Point
# ============================================================================

@mcp.tool()
async def generate_cbt_protocol(
    user_intent: str,
    max_iterations: int = 5,
    wait_for_approval: bool = True
) -> str:
    """
    Generate a comprehensive CBT protocol using the multi-agent LangGraph workflow.
    
    This is the PRIMARY tool that exposes your entire complex workflow as a single action.
    
    The workflow includes:
    - Drafter Agent: Creates evidence-based CBT protocols
    - Safety Guardian: Validates clinical safety
    - Clinical Critic: Ensures therapeutic quality
    - Supervisor: Orchestrates the multi-agent system
    
    The workflow runs iteratively until quality threshold (≥7.5/10) is met or max_iterations reached.
    
    Args:
        user_intent: Clinical intent for the protocol. Examples:
            - 'Create an exposure hierarchy for social anxiety with public speaking focus'
            - 'Develop a sleep hygiene protocol for adult insomnia patients'
            - 'Design cognitive restructuring exercises for adolescent depression'
            - 'Build a behavioral activation plan for post-partum depression'
        
        max_iterations: Maximum revision iterations (1-10, default: 5)
        
        wait_for_approval: If True (default), returns the protocol when ready for review.
                          If False, returns immediately with thread_id for async tracking.
    
    Returns:
        If wait_for_approval=True (default):
            - Waits for workflow to reach human review stage
            - Returns the complete generated protocol
        
        If wait_for_approval=False:
            - Thread ID and status for async monitoring
            - Use Resources to view protocol details
    
    Use Case (from task):
        User: "Ask Cerina Foundry to create a sleep hygiene protocol"
        → This triggers the backend, runs agents, returns result
        → Bypasses the React UI but uses the same underlying logic
    """
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            # Start the workflow
            print(f"[MCP] Starting workflow for: {user_intent}")
            response = await client.post(
                f"{API_BASE_URL}/generate",
                json={
                    "user_intent": user_intent,
                    "max_iterations": max_iterations
                }
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                return f"❌ **Error:** {error_data.get('detail', 'Unknown error')}"
            
            data = response.json()
            thread_id = data['thread_id']
            print(f"[MCP] Workflow started with thread_id: {thread_id}")
            
            # Track this thread for resources
            if thread_id not in _active_threads:
                _active_threads.append(thread_id)
            
            # If not waiting for approval, return immediately
            if not wait_for_approval:
                return f"""✅ **CBT Protocol Workflow Started**

**Thread ID:** `{thread_id}`
**Status:** {data['status']}

**🔄 Multi-Agent System Active:**
The LangGraph workflow is processing your request:
1. 📝 **Drafter** - Creating protocol
2. 🛡️ **Safety Guardian** - Validating safety
3. ⭐ **Clinical Critic** - Reviewing quality
4. 👔 **Supervisor** - Orchestrating workflow

**📊 What's Happening:**
- The agents will iterate up to {max_iterations} times
- Each iteration improves quality based on critic feedback
- Safety validation happens at every step
- Workflow continues until quality ≥7.5/10

**📄 Access Your Protocol:**
Use the MCP Resource: `cerina://protocol/{thread_id}` to view:
- Current status and progress
- Protocol draft (live updates)
- Quality metrics
- Safety validations

**✅ Human Review:**
When the workflow completes, the protocol will be ready for review.
Access the full draft via the resource endpoint.

💡 **Tip:** The workflow runs asynchronously. Check the resource for real-time updates!
"""
            
            # ✅ FIXED: Poll with early exit for pending_human_review
            else:
                max_wait = 600  # 10 minutes timeout
                poll_interval = 5
                elapsed = 0
                poll_count = 0
                
                while elapsed < max_wait:
                    # Add small delay before polling (except first time)
                    if poll_count > 0:
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                    
                    poll_count += 1
                    print(f"[MCP] Polling status (attempt {poll_count}, elapsed: {elapsed}s)")
                    
                    # Check status
                    try:
                        status_resp = await client.get(
                            f"{API_BASE_URL}/state/{thread_id}",
                            timeout=30.0
                        )
                    except httpx.TimeoutException:
                        print(f"[MCP] Status check timeout on attempt {poll_count}")
                        continue
                    
                    if status_resp.status_code != 200:
                        return f"❌ Error checking status: {status_resp.status_code}"
                    
                    status_data = status_resp.json()
                    approval_status = status_data.get('approval_status')
                    iteration = status_data.get('iteration_count', 0)
                    
                    print(f"[MCP] Status: {approval_status}, Iteration: {iteration}")
                    
                    # ✅ KEY FIX: Check for pending_human_review status
                    if approval_status == 'pending_human_review':
                        print(f"[MCP] Protocol ready for review - returning response")
                        
                        # Extract the protocol draft
                        current_draft = status_data.get('current_draft', 'No draft available')
                        safety_flags = status_data.get('safety_flags_count', 0)
                        quality_reviews = status_data.get('critic_feedbacks_count', 0)
                        has_issues = status_data.get('has_blocking_issues', False)
                        
                        return f"""# ✅ CBT Protocol Generated Successfully

**Thread ID:** `{thread_id}`
**Status:** Ready for Human Review
**Iterations Completed:** {iteration}/{max_iterations}

---

## 📊 Quality Metrics

- **Safety Validations:** {safety_flags} flag(s) reviewed
- **Quality Reviews:** {quality_reviews} review(s) completed
- **Blocking Issues:** {'⚠️ Yes - Requires attention' if has_issues else '✅ None'}

---

## 📄 Generated Protocol

{current_draft}

---

## 🔄 Multi-Agent Review Summary

This protocol was generated through our multi-agent system:

1. **📝 CBT Drafter** - Created evidence-based protocol structure
2. **🛡️ Safety Guardian** - Validated clinical safety and contraindications
3. **⭐ Clinical Critic** - Assessed therapeutic quality and empathy
4. **👔 Supervisor** - Orchestrated workflow and quality control

---

## ℹ️ Next Steps

**To Approve:**
POST {API_BASE_URL}/resume/{thread_id}
Body: {{"action": "approve", "thread_id": "{thread_id}"}}

text

**To Request Revisions:**
POST {API_BASE_URL}/resume/{thread_id}
Body: {{"action": "reject", "feedback": "Your feedback here", "thread_id": "{thread_id}"}}

text

**To Edit and Approve:**
POST {API_BASE_URL}/resume/{thread_id}
Body: {{"action": "edit", "edited_draft": "Your edited version", "thread_id": "{thread_id}"}}

text

---

💡 **Note:** This protocol is pending human review. It has passed automated quality checks but should be reviewed by a qualified clinician before clinical use.

**Created:** {status_data.get('created_at', 'N/A')}
**Resource URI:** `cerina://protocol/{thread_id}`
"""
                    
                    elif approval_status == 'approved':
                        # Protocol was approved
                        print(f"[MCP] Protocol approved - returning final version")
                        final_draft = status_data.get('final_approved_draft') or status_data.get('current_draft', 'No draft available')
                        
                        return f"""# ✅ CBT Protocol - APPROVED & FINALIZED

**Thread ID:** `{thread_id}`
**Status:** ✅ Approved and Ready for Clinical Use
**Iterations:** {iteration}

---

## 📄 FINAL APPROVED PROTOCOL

{final_draft}

---

## ✅ Validation Summary

- **Safety Guardian:** Passed
- **Clinical Critic:** Quality threshold met
- **Human Reviewer:** Approved

**Approved At:** {status_data.get('approved_at', 'N/A')}

---

🎉 **This protocol is finalized and ready for clinical implementation.**
"""
                    
                    elif approval_status == 'rejected':
                        print(f"[MCP] Protocol rejected")
                        return f"❌ **Protocol Rejected**\n\nThe protocol was rejected during review.\nThread ID: `{thread_id}`\n\nView details: `cerina://protocol/{thread_id}`"
                    
                    elif approval_status in ['failed', 'error']:
                        print(f"[MCP] Workflow error: {approval_status}")
                        return f"❌ **Workflow Error**\n\nThe workflow encountered an error.\nThread ID: `{thread_id}`\nStatus: {approval_status}"
                    
                    # Still in progress - show progress
                    print(f"[MCP] Still in progress: {approval_status}")
                
                # Timeout - return what we have
                print(f"[MCP] Reached timeout ({max_wait}s) - fetching final state")
                try:
                    status_resp = await client.get(f"{API_BASE_URL}/state/{thread_id}", timeout=30.0)
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        current_draft = status_data.get('current_draft', 'Still generating...')
                        
                        return f"""⏱️ **Workflow Still In Progress**

**Thread ID:** `{thread_id}`
**Status:** {status_data.get('approval_status', 'unknown')}
**Time Elapsed:** {max_wait}s
**Polled:** {poll_count} times

The workflow is taking longer than expected, but it's still running.

**Current Progress:**
- Iteration: {status_data.get('iteration_count', 0)}/{max_iterations}
- Safety Checks: {status_data.get('safety_flags_count', 0)}
- Quality Reviews: {status_data.get('critic_feedbacks_count', 0)}

**Partial Draft (if available):**

{current_draft if current_draft != 'Still generating...' else 'Draft generation in progress...'}

---

💡 **Check progress:** Use resource `cerina://protocol/{thread_id}` to view real-time updates.
"""
                    else:
                        return f"⏱️ **Timeout:** Workflow exceeded {max_wait}s.\n\nThread ID: `{thread_id}`\n\nUse `cerina://protocol/{thread_id}` to check progress."
                except Exception as e:
                    return f"⏱️ **Timeout:** Workflow exceeded {max_wait}s.\n\nThread ID: `{thread_id}`\n\nError: {str(e)}"
        
        except httpx.ConnectError as e:
            print(f"[MCP] Connection error: {str(e)}")
            return "🔌 **Connection Error:** FastAPI server not running at http://localhost:8000\n\nPlease start the server with: `python main.py`"
        except httpx.TimeoutException as e:
            print(f"[MCP] Timeout error: {str(e)}")
            return "⏱️ **Timeout:** The workflow is taking longer than expected.\n\nThis might be normal for complex protocols. Try increasing max_iterations or check server logs."
        except Exception as e:
            print(f"[MCP] Unexpected error: {str(e)}")
            return f"❌ **Unexpected Error:** {str(e)}\n\nPlease check server logs for details."


# ============================================================================
# RESOURCES: Read-Only Data Access (Not Actions)
# ============================================================================

@mcp.resource("cerina://protocol/{thread_id}")
async def get_protocol(thread_id: str) -> str:
    """
    Resource endpoint to view a protocol's current state.
    
    This is NOT a tool (action), but a RESOURCE (data).
    It provides read-only access to protocol status and content.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Get full state
            response = await client.get(f"{API_BASE_URL}/state/{thread_id}")
            
            if response.status_code != 200:
                return f"❌ Protocol not found: {thread_id}"
            
            data = response.json()
            
            status_emoji = {
                "pending_human_review": "⏸️ Awaiting Human Review",
                "approved": "✅ Approved",
                "rejected": "❌ Rejected",
                "in_progress": "🔄 In Progress",
                "draft": "📝 Drafting",
                "validating": "🛡️ Safety Check",
                "reviewing": "⭐ Quality Review"
            }.get(data.get('approval_status', ''), "📊 Unknown")
            
            draft = data.get('current_draft') or data.get('final_approved_draft', 'No draft available yet')
            
            return f"""# Protocol: {thread_id}

## Status: {status_emoji}

**Progress:** Iteration {data.get('iteration_count', 0)}/{data.get('max_iterations', 0)}

---

## 📈 Quality Metrics

- **Safety Flags:** {data.get('safety_flags_count', 0)}
- **Quality Reviews:** {data.get('critic_feedbacks_count', 0)}
- **Blocking Issues:** {'Yes ⚠️' if data.get('has_blocking_issues') else 'No ✅'}
- **Finalized:** {'Yes ✓' if data.get('is_finalized') else 'No'}

---

## 📄 Current Draft

{draft}

---

## ℹ️ Workflow Info

This protocol was generated by the Cerina multi-agent system:
- **Drafter:** Evidence-based protocol creation
- **Safety Guardian:** Clinical safety validation
- **Clinical Critic:** Quality assurance
- **Supervisor:** Workflow orchestration

**User Intent:** {data.get('user_intent', 'N/A')}
**Created:** {data.get('created_at', 'N/A')}
**Last Modified:** {data.get('last_modified', 'N/A')}
"""
        
        except Exception as e:
            return f"❌ Error accessing protocol: {str(e)}"


@mcp.resource("cerina://protocols")
async def list_protocols() -> str:
    """
    List all active protocols.
    """
    if not _active_threads:
        return "No active protocols. Use `generate_cbt_protocol` to create one!"
    
    result = "# Active Protocols\n\n"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for thread_id in _active_threads:
            try:
                response = await client.get(f"{API_BASE_URL}/state/{thread_id}")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('approval_status', 'unknown')
                    intent = data.get('user_intent', 'No description')[:60]
                    result += f"- `{thread_id}`: **{status}** - {intent}...\n"
                else:
                    result += f"- `{thread_id}`: (unavailable)\n"
            except:
                result += f"- `{thread_id}`: (error)\n"
    
    result += f"\n\n💡 **Tip:** Access any protocol with `cerina://protocol/{{thread_id}}`"
    return result