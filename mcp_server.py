"""
MCP Server for Cerina Protocol Foundry using FastMCP.

Exposes the LangGraph CBT protocol workflow as:
- 1 PRIMARY TOOL: generate_cbt_protocol (the entire workflow)
- RESOURCES: For viewing generated protocols (read-only data)
"""

from mcp.server.fastmcp import FastMCP
import httpx
from typing import Optional, List
import asyncio
import time
import sys  # <--- REQUIRED FOR SAFE LOGGING

mcp = FastMCP("Cerina Protocol Foundry")

API_BASE_URL = "http://localhost:8000/api"

# Store active thread IDs for resource discovery
_active_threads: List[str] = []


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
    
    This tool exposes the entire backend workflow through MCP.
    
    When wait_for_approval=True (default):
    - The workflow automatically bypasses the human review halt
    - Returns the finalized protocol directly
    - Perfect for automated/programmatic use
    
    When wait_for_approval=False:
    - Returns thread_id immediately
    - Allows async tracking via resources
    """
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            # ✅ LOGGING FIX: Send to stderr to avoid breaking JSON
            print(f"[MCP] Starting workflow for: {user_intent}", file=sys.stderr)
            
            response = await client.post(
                f"{API_BASE_URL}/generate",
                json={
                    "user_intent": user_intent,
                    "max_iterations": max_iterations,
                    "source": "mcp" 
                }
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.text else {}
                return f"❌ **Error:** {error_data.get('detail', 'Unknown error')}"
            
            data = response.json()
            thread_id = data['thread_id']
            print(f"[MCP] Workflow started with thread_id: {thread_id}", file=sys.stderr)
            
            # Track this thread for resources
            if thread_id not in _active_threads:
                _active_threads.append(thread_id)
            
            # If not waiting, return immediately
            if not wait_for_approval:
                return f"""✅ **CBT Protocol Workflow Started**

**Thread ID:** `{thread_id}`
**Status:** {data['status']}
**Mode:** Auto-finalize (MCP bypass mode)

The workflow will complete automatically without human review halt.
Access the final protocol via: `cerina://protocol/{thread_id}`
"""
            
            # ✅ Wait for finalization (should be quick with bypass mode)
            else:
                max_wait = 300  # 5 minutes should be plenty
                poll_interval = 3
                elapsed = 0
                poll_count = 0
                
                while elapsed < max_wait:
                    if poll_count > 0:
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                    
                    poll_count += 1
                    print(f"[MCP] Polling status (attempt {poll_count}, elapsed: {elapsed}s)", file=sys.stderr)
                    
                    try:
                        status_resp = await client.get(
                            f"{API_BASE_URL}/state/{thread_id}",
                            timeout=30.0
                        )
                    except httpx.TimeoutException:
                        print(f"[MCP] Status check timeout on attempt {poll_count}", file=sys.stderr)
                        continue
                    
                    if status_resp.status_code != 200:
                        return f"❌ Error checking status: {status_resp.status_code}"
                    
                    status_data = status_resp.json()
                    approval_status = status_data.get('approval_status')
                    iteration = status_data.get('iteration_count', 0)
                    is_finalized = status_data.get('is_finalized', False)
                    
                    print(f"[MCP] Status: {approval_status}, Finalized: {is_finalized}, Iteration: {iteration}", file=sys.stderr)
                    
                    # ✅ Check if finalized (should happen automatically with bypass mode)
                    if is_finalized or approval_status == 'approved':
                        print(f"[MCP] Protocol finalized - returning response", file=sys.stderr)
                        
                        final_draft = status_data.get('final_approved_draft') or status_data.get('current_draft', 'No draft available')
                        safety_flags = status_data.get('safety_flags_count', 0)
                        quality_reviews = status_data.get('critic_feedbacks_count', 0)
                        
                        return f"""# ✅ CBT Protocol Generated Successfully

**Thread ID:** `{thread_id}`
**Status:** ✅ Auto-Finalized (MCP Mode)
**Iterations Completed:** {iteration}/{max_iterations}

---

## 📊 Quality Metrics

- **Safety Validations:** {safety_flags} flag(s) reviewed
- **Quality Reviews:** {quality_reviews} review(s) completed
- **Mode:** Automated (bypassed human review)

---

## 📄 Generated Protocol

{final_draft}

---

## 🔄 Multi-Agent Review Summary

This protocol was generated through our multi-agent system:

1. **📝 CBT Drafter** - Created evidence-based protocol structure
2. **🛡️ Safety Guardian** - Validated clinical safety and contraindications
3. **⭐ Clinical Critic** - Assessed therapeutic quality and empathy
4. **👔 Supervisor** - Orchestrated workflow and quality control

---

✅ **This protocol was automatically finalized for MCP usage.**

**Created:** {status_data.get('created_at', 'N/A')}
**Resource URI:** `cerina://protocol/{thread_id}`
"""
                    
                    # Handle failures
                    elif approval_status in ['failed', 'error']:
                        print(f"[MCP] Workflow error: {approval_status}", file=sys.stderr)
                        return f"❌ **Workflow Error**\n\nStatus: {approval_status}\nThread ID: `{thread_id}`"

                    # Handle case where it halts anyway (fallback)
                    elif approval_status == 'pending_human_review':
                         print(f"[MCP] Workflow halted (unexpected in bypass mode) - returning current draft", file=sys.stderr)
                         current_draft = status_data.get('current_draft', 'No draft available')
                         return f"""# ⏸️ Protocol Ready for Review
                         
**Thread ID:** `{thread_id}`
**Status:** Pending Review

The system halted for manual review. You can approve it via the API or view the draft below.

---
{current_draft}
"""
                    
                    # Still in progress
                    print(f"[MCP] Still in progress: {approval_status}", file=sys.stderr)
                
                # Timeout
                print(f"[MCP] Reached timeout ({max_wait}s)", file=sys.stderr)
                return f"⏱️ **Timeout:** Workflow exceeded {max_wait}s.\n\nThread ID: `{thread_id}`\n\nCheck: `cerina://protocol/{thread_id}`"
        
        except httpx.ConnectError as e:
            print(f"[MCP] Connection error: {str(e)}", file=sys.stderr)
            return "🔌 **Connection Error:** FastAPI server not running at http://localhost:8000"
        except Exception as e:
            print(f"[MCP] Unexpected error: {str(e)}", file=sys.stderr)
            return f"❌ **Unexpected Error:** {str(e)}"


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